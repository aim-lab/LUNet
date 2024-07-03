import tensorflow as tf
from tensorflow.keras import layers as KL
from tensorflow.keras import backend as K
import keras

##Reimplementation of the soft skeletonize from the centerline dice loss in tensorflow
def soft_skeletonize(x, thresh_width=10):
    """
    Differenciable aproximation of morphological skelitonization operaton
    thresh_width - needs to be greater then or equal to the maximum radius for the tube-like structure
    """

    minpool = (
        lambda y: K.pool2d(
            y * -1,
            pool_size=(3, 3),
            strides=(1, 1),
            pool_mode="max",
            data_format="channels_last",
            padding="same",
        )
        * -1
    )
    maxpool = lambda y: K.pool2d(
        y,
        pool_size=(3, 3),
        strides=(1, 1),
        pool_mode="max",
        data_format="channels_last",
        padding="same",
    )

    for i in range(thresh_width):
        min_pool_x = minpool(x)
        contour = K.relu(maxpool(min_pool_x) - min_pool_x)
        x = K.relu(x - contour)
    return x


def norm_intersection(center_line, vessel):
    """
    inputs shape  (batch, channel, height, width)
    intersection formalized by first ares
    x - suppose to be centerline of vessel (pred or gt) and y - is vessel (pred or gt)
    """
    smooth = 1.0
    clf = tf.reshape(
        center_line, (tf.shape(center_line)[0], tf.shape(center_line)[-1], -1)
    )
    vf = tf.reshape(vessel, (tf.shape(vessel)[0], tf.shape(vessel)[-1], -1))
    intersection = K.sum(clf * vf, axis=-1)
    return (intersection + smooth) / (K.sum(clf, axis=-1) + smooth)

## Centerline dice loss
def soft_cldice_loss(k=10, data_format="channels_last"):
    """clDice loss function for tensorflow/keras
    Args:
        k: needs to be greater or equal to the maximum radius of the tube structure.
        data_format: either channels_first or channels_last        
    Returns:
        loss_function(y_true, y_pred)  
    """

    def loss(target, pred):
        cl_pred = soft_skeletonize(pred, thresh_width=k)
        target_skeleton = soft_skeletonize(target, thresh_width=k)
        iflat = norm_intersection(cl_pred, target)
        tflat = norm_intersection(target_skeleton, pred)
        intersection = iflat * tflat
        return tf.reduce_mean(1 - ((2.0 * intersection) / (iflat + tflat)),axis = -1)

    return loss

## Dice loss
@tf.function
def dice_loss(y_true, y_pred):
    y_pred = tf.math.sigmoid(y_pred)
    numerator = tf.reduce_sum(tf.math.multiply(y_true,y_pred),axis = (1,2,3))
    denominator = tf.reduce_sum(tf.math.add(y_true,y_pred),axis = (1,2,3))
    dice = 1 - (2*numerator) / (denominator)
    return dice

## Cross entropy loss
@tf.function
def cross_entropy_loss(y_true,y_pred):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(y_true, y_pred) ,axis = (1,2,3))

## Total variation loss
@tf.function
def TV_Loss(y_pred):
    y = y_pred[:,:,1:,:] - y_pred[:,:,:-1,:] # horizontal and vertical directions
    x = y_pred[:,1:,:,:] - y_pred[:,:-1,:,:]
    delta_x = x[:,1:,:-2,:]
    delta_y = y[:,:-2,1:,:]
    delta_u = tf.math.abs(delta_x) + tf.math.abs(delta_y)
    lenth = tf.reduce_mean(delta_u + 0.00000001,axis = (1,2,3)) # equ.(11) in the paper
    return lenth

##Define lunet loss for a single channel
soft_ci = soft_cldice_loss()
@tf.function
def lunet_loss_(y_true,y_pred):
    return dice_loss(y_true, y_pred) + cross_entropy_loss(y_true,y_pred) + TV_Loss(tf.math.sigmoid(y_pred)) + 0.3*soft_ci(y_true, tf.math.sigmoid(y_pred))

##Compute lunet loss for each channel: artery, veins, and general bv (artery + veins + unknown)
@tf.function
def lunet_loss(y_true,y_pred):
    y_true0 = y_true[:,:,:,:1]
    y_true1 = y_true[:,:,:,1:2]
    y_pred0 = y_pred[:,:,:,:1]
    y_pred1 = y_pred[:,:,:,1:2]
    y_true2 = y_true[:,:,:,2:3]
    y_pred2 = y_pred[:,:,:,2:3]

    return lunet_loss_(y_true0,y_pred0) + lunet_loss_(y_true1,y_pred1) + lunet_loss_(y_true2,y_pred2)

##Embed lunet loss within a custom wrapper for multi gpu stability
class CustomLoss(keras.__internal__.losses.LossFunctionWrapper):
    def __init__(
        self,
        function,
        reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
        name="loss",
    ):
        super().__init__(
            function,
            name=name,
            reduction=reduction,
        )