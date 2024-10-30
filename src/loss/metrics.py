import tensorflow as tf

## Computation of the dice score for the 1 pixel vs the non 1 pixel within a tensor image of size bs,h,w,1
def dice_1(y_true,y_pred,smooth = 1e-10):
    k=1
    y_pred = tf.round(tf.sigmoid(y_pred))
    intersection = tf.reduce_sum(tf.cast(tf.math.logical_and(y_true == k,y_pred == k),tf.float32),axis = [1,2,3])
    union_plus_intersection = tf.reduce_sum(tf.cast(y_true == k,tf.float32),axis = [1,2,3]) + tf.reduce_sum(tf.cast(y_pred == k,tf.float32),axis = [1,2,3])
    dice = tf.reduce_mean((2*intersection + smooth) / (union_plus_intersection + smooth))
    return dice

## Extraction of the first channel then computation of the dice score of this specific channel
def dice_1_0(y_true,y_pred,smooth = 1e-10,unknown = None):
    y_true0 = y_true[:,:,:,:1]
    y_pred0 = y_pred[:,:,:,:1]
    if unknown is not None:
        y_true0 = y_true0*(1-unknown)
        y_pred0 = y_pred0*(1-unknown)
    return dice_1(y_true0,y_pred0) 

## Extraction of the second channel then computation of the dice score of this specific channel
def dice_1_1(y_true,y_pred,smooth = 1e-10, unknown = None):
    y_true1 = y_true[:,:,:,1:2]
    y_pred1 = y_pred[:,:,:,1:2]
    if unknown is not None:
        y_true1 = y_true1*(1-unknown)
        y_pred1 = y_pred1*(1-unknown)
    return dice_1(y_true1,y_pred1) 

## Extraction of the third channel then computation of the dice score of this specific channel
def dice_1_2(y_true,y_pred,smooth = 1e-10,unknown = None):
    y_true1 = y_true[:,:,:,2:3]
    y_pred1 = y_pred[:,:,:,2:3]
    if unknown is not None:
        y_true1 = y_true1*(1-unknown)
        y_pred1 = y_pred1*(1-unknown)
    return dice_1(y_true1,y_pred1) 