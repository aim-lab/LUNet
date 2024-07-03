import argparse
##Loading args which include the path of the dataset root and the name of the save model
parser = argparse.ArgumentParser()
parser.add_argument("path", help="path of the dataset root",)
parser.add_argument("model_name", help = "name of the saved model",)
parser.add_argument("verbose", type=int, default=2, help="Set the verbosity level (default: 2)")

args = parser.parse_args()
Path = args.path
model_name = args.model_name
verbose = args.verbose
print('*'*50)
print('Starting experiment :' + Path)
import os
print(os.listdir())

##Importing the necessary libraries

#install(["tensorflow-addons", "tqdm","albumentations==1.2.1"], quietly=True)
import tensorflow as tf
from tensorflow.keras.layers import *
import random
import numpy as np
from src.data.datasets import get_datasets
from src.lunet.model import *
from src.loss.loss import *
from src.loss.metrics import *

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

##Train-val-test tensorflow dataset creation
DATASETS_TRAIN = ["UZLF_TRAIN"]
DATASETS_VAL = ["UZLF_VAL"]
DATASETS_TEST = ["UZLF_TEST"]

train_list = [Path + element + "/images" for element in DATASETS_TRAIN]
val_list = [Path + element + "/images" for element in DATASETS_VAL]
test_list = [Path + element + "/images" for element in DATASETS_TEST]

AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds_tf,val_ds_tf, test_ds_tf = get_datasets(train_list,val_list,test_list, AUTOTUNE)

#Creating another callback to save the best model according to the validation loss
best_model = model_name + 'best.h5'
checkpoint = tf.keras.callbacks.ModelCheckpoint(best_model, monitor="val_loss", mode="min",save_best_only=True, verbose=1,save_weights_only=True)

#Printing the number of detected gpus
print(tf.config.list_physical_devices())
mirrored_strategy = tf.distribute.MirroredStrategy()
print("*"*50)
print('Number of devices: {}'.format(mirrored_strategy.num_replicas_in_sync))

#model instanciation with loss and metrics follow up
with mirrored_strategy.scope():    
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
                   metrics=[dice_1_0,dice_1_1,dice_1_2]
                   )

#Batch size per gpu definition
BATCH_SIZE_PER_REPLICA = 1
#Batch size globaldefinition
BATCH_SIZE = BATCH_SIZE_PER_REPLICA * mirrored_strategy.num_replicas_in_sync

#final dataset definition including the batch size
train_ds = train_ds_tf.shuffle(len(train_ds_tf)).batch(BATCH_SIZE).prefetch(AUTOTUNE)
val_ds = val_ds_tf.batch(BATCH_SIZE).prefetch(AUTOTUNE)
test_ds = test_ds_tf.batch(BATCH_SIZE).prefetch(AUTOTUNE)

#model fitting
history = my_model.fit(train_ds,
                         epochs=1300,
                         validation_data=val_ds, 
                         callbacks = [checkpoint],
                         verbose = verbose
                        )

#tf history saving in case you may want to consult the loss and metrics during training
np.save('my_historyfinal.npy',history.history)






