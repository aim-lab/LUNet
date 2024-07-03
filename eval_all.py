import argparse
##Loading args which include the path of the dataset root and the name of the save model
parser = argparse.ArgumentParser()
parser.add_argument("path", help="path of the dataset root",)
parser.add_argument("model_name", help = "name of the saved model",)
parser.add_argument("--use_TTDA", help = "using TTDA or not", type=bool, default=True)
args = parser.parse_args()
Path = args.path
model_name = args.model_name
use_TTDA = args.use_TTDA
print('*'*50)
print('Starting experiment :' + Path)

##Importing the necessary libraries

#install(["tensorflow-addons", "tqdm","pillow"], quietly=True)
from PIL import Image
import tensorflow as tf
from tensorflow.keras.layers import *
import random
import tensorflow_addons as tfa
import math
import numpy as np
from src.data.datasets import get_datasets
from src.lunet.model import *
from src.loss.loss import *
from src.loss.metrics import *
import os

##Making the experience deterministic for reproducibility
os.environ['PYTHONHASHSEED'] = "0"
random.seed(0)
np.random.seed(0)
tf.random.set_seed(0)
os.environ['TF_DETERMINISTIC_OPS'] = "1"
tf.keras.utils.set_random_seed(0)
tf.config.experimental.enable_op_determinism()

##Shape on which the images will be processed by the model
final_shape = 1472

##Test tensorflow dataset creation
DATASETS_TEST = ["UZLF_VAL", "UZLF_TEST"]

test_list = [Path + element + "/images" for element in DATASETS_TEST]

print("*"*50)
##Initialising model
inputs = tf.keras.layers.Input((final_shape,final_shape,3))
init_n_filters = 18
kernel_size = 7
attention_kernel_size = 14
transpose_stride = 2
stride = 1
keep_prob = 0.9
block_size = 7
drop_block = True
with_batch_normalization = True
my_model, attention_a_unet,bottleneck = build_lunet(inputs,init_n_filters,kernel_size,transpose_stride,stride,keep_prob,block_size,with_batch_normalization = with_batch_normalization, scale = True,dropblock = drop_block)
my_model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                 loss=CustomLoss(lunet_loss),
                 #metrics=[dice_1_0,dice_1_1,dice_1_2]
                 )
##loading model weights
my_model.load_weights( model_name + 'best.h5')

##Single image sample prediction using test time data augmentation (TTDA)
def make_pred(batch_x):
    ##(TTDA) parameters
    angles = [i for i in range(0, 360, 90)]
    transpose = [False, True]
    #make a prediction for each angle with and without image transposition and save it in the
    preds = []
    for angle in angles:
      for t in transpose:
          x_ = tfa.image.rotate(images=batch_x, angles=angle * math.pi / 180)
          if t:
              x_ = tf.image.transpose(x_)
          pred_ = my_model(x_, training=False)

          if t:
              pred_ = tf.image.transpose(pred_)
          pred_ = tfa.image.rotate(images=pred_, angles=(360 - angle) * math.pi / 180)

          preds.append(pred_)
    #agregatting the segmentation across all predictions
    preds = tf.stack(preds)
    final_pred = tf.reduce_mean(preds, axis=0)
    return final_pred

#For each test dataset
for dataset in test_list:
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    #Create a dataset folder within the prediction folder
    dataset_name = dataset.split('/')[1]
    os.makedirs("predictions/"+dataset_name,exist_ok=True)

    #loading the dataset
    train_ds_tf, val_ds_tf, test_ds_tf = get_datasets(None, None, [dataset], AUTOTUNE, with_name = True)
    test_ds = test_ds_tf.batch(1).prefetch(AUTOTUNE)

    #Initialising A/V/and general blood vessel score arrays
    veins_dice = []
    artery_dice = []
    global_ = []

    #For each sample in the dataset
    for x,out, name, orig_size in test_ds:
        # Make a prediction with test time data augmentation if asked, normal prediction otherwise
        if use_TTDA == True:
            pred = make_pred(x) * tf.cast(x[:,:,:,:1]>0, tf.float32)
        else:
            pred = my_model(x, training=False)
        pred = pred * x[:,:,:,:1]
        #Remove the padding added to the image
        pred = tf.image.resize(pred[:,14:-14,14:-14],(orig_size[0,0], orig_size[0,1]))
        #Storing the pred into a RGBA image prediction
        prediction = np.zeros((orig_size[0,0], orig_size[0,1],4))
        prediction[:,:,0] = 255*(np.array(pred)[0,:,:,0] > 0)
        prediction[:,:,2] = 255*(np.array(pred)[0,:,:,1] > 0)
        prediction[:,:,3] = tf.maximum(prediction[:,:,0],prediction[:,:,2])
        # Convert the prediction tensor to a numpy array
        prediction_np = prediction.astype(np.uint8)
        # Convert the numpy array to a PIL image
        prediction_image = Image.fromarray(prediction_np, 'RGBA')
        # Save the image
        file_name = str(name.numpy())[:-2]
        base_name = str(os.path.basename(file_name)).replace('jpg','png')
        dataset_name = dataset.split('/')[1]
        prediction_image.save(f"Pred_1_all_data/{dataset_name}/{base_name}")