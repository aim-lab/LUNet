from .data_utils import *
import tensorflow as tf
import os

## Take as input a single list of dataset paths and if it is a training dataset and return it as a tf dataset
def get_dataset(ds_list,is_train, AUTOTUNE,jitter = True,with_name = False, with_output = True):
  # Define the list of file extensions you want to include
  file_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
  
  # Initialize an empty list to store file paths
  all_files = []
  for directory in ds_list:
    # Loop through each file extension and gather the files
    for ext in file_extensions:
        all_files.extend(tf.io.gfile.glob(os.path.join(directory, ext)))
        
  img = tf.data.Dataset.list_files(all_files,shuffle = True)
  if is_train:
    if jitter == True:
      ds_tf = img.map(parse_image_tf, num_parallel_calls=AUTOTUNE)
    else:
      ds_tf = img.map(parse_image_tf_nojitter, num_parallel_calls=AUTOTUNE)
  else:
    if with_name:
      if with_output:
        ds_tf = img.map(parse_image_tf_test_with_name, num_parallel_calls=AUTOTUNE)
      else:
        ds_tf = img.map(parse_image_tf_test_with_name_without_output, num_parallel_calls=AUTOTUNE)
    else:
      ds_tf = img.map(parse_image_tf_test, num_parallel_calls=AUTOTUNE)
  return ds_tf


##Take as input a list datasets path to use in train, val and test and return them as tf datasets
def get_datasets(train_list,val_list,test_list, AUTOTUNE, jitter = True,with_name = False, with_output = True):
  if train_list is not None:
    train_ds_tf = get_dataset(train_list,True,AUTOTUNE,jitter = jitter)
  else:
    train_ds_tf = None
  
  if val_list is not None:
    val_ds_tf = get_dataset(val_list,False,AUTOTUNE)
  else:
    val_ds_tf = None
    
  if test_list is not None:
    test_ds_tf = get_dataset(test_list,False,AUTOTUNE,with_name = with_name,with_output=with_output)
  else:
    test_ds_tf = None
    
  return train_ds_tf,val_ds_tf,test_ds_tf
  
  
  