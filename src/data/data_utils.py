import tensorflow as tf
import tensorflow_addons as tfa
import math
def padding(x,y,z):
    x = tf.image.pad_to_bounding_box(x, 14, 14, 1472, 1472) 
    y = tf.image.pad_to_bounding_box(y, 14, 14, 1472, 1472)
    z = tf.image.pad_to_bounding_box(z, 14, 14, 1472, 1472)
    return x,y,z

def normalize(x, y,z):
    x = tf.cast(x, tf.float32) / 255.0
    y = tf.cast(y,tf.float32)/255.0
    z = tf.cast(z,tf.float32)/255.0
    return x, y,z
    
# Define the brightness, contrast, and saturation jitter values
brightness_jitter = (0.1, 0.2)
contrast_jitter = (0.8, 3)
saturation_jitter = (0.4, 2)
# Define the hue jitter value (fixed at 0 in this example)
hue_jitter_value = 0

# Define the probability of applying the jitter
jitter_probability = 0.2

def apply_jitter_ops(image,brightness_jitter_value,contrast_jitter_value,saturation_jitter_value,hue_jitter_value):
  image = tf.image.adjust_brightness(image, delta=brightness_jitter_value)
  image = tf.image.adjust_contrast(image, contrast_factor=contrast_jitter_value)
  image = tf.image.adjust_saturation(image, saturation_factor=saturation_jitter_value)
  #image = tf.image.adjust_hue(image, delta=hue_jitter_value)
  return image
  
def parse_image_tf_nojitter(x):
    y = tf.strings.regex_replace(x,"images", "veins")
    z = tf.strings.regex_replace(x,"images", "artery")
    x = tf.io.read_file(x)
    x = tf.io.decode_image(x,channels = 3,expand_animations=False)
    
    y = tf.io.read_file(y)
    y = tf.io.decode_image(y, channels = 1,expand_animations=False)
    
    z = tf.io.read_file(z)
    z = tf.io.decode_image(z, channels = 1,expand_animations=False)
    
    x = tf.image.resize(x,(1444,1444))
    y = tf.image.resize(y,(1444,1444),method = "nearest")
    z = tf.image.resize(z,(1444,1444),method = "nearest")
    
    x,y,z = padding(x,y,z)
    
    x_orig = x
    mask = tf.cast(x_orig[:,:,:1]>0,tf.float32)
    
    # color jitter
    brightness_jitter_value = tf.random.uniform(shape=[], minval=brightness_jitter[0], maxval=brightness_jitter[1])
    contrast_jitter_value = tf.random.uniform(shape=[], minval=contrast_jitter[0], maxval=contrast_jitter[1])
    saturation_jitter_value = tf.random.uniform(shape=[], minval=saturation_jitter[0], maxval=saturation_jitter[1])
    # Define a random variable that determines whether or not to apply the jitter
    apply_jitter =  tf.less(tf.random.uniform(shape=[], minval=0, maxval=1),jitter_probability)
    x = tf.cond(apply_jitter, lambda: apply_jitter_ops(x,brightness_jitter_value,contrast_jitter_value,saturation_jitter_value,hue_jitter_value), lambda: x)
    x = tf.convert_to_tensor(x) *mask
    y = tf.cast(y, tf.float32) * mask
    z = tf.cast(z, tf.float32) * mask
    
    angle = tf.random.uniform(shape=[], minval=0, maxval=360, dtype=tf.float32)
    x = tfa.image.rotate(images = x,angles = angle * math.pi / 180)
    y = tfa.image.rotate(images = y,angles = angle * math.pi / 180)
    z = tfa.image.rotate(images = z,angles = angle * math.pi / 180)
    #x = tf.image.rot90(x, k=k)
    #y = tf.image.rot90(y, k=k)
    #z = tf.image.rot90(z, k=k)
    #u = tf.image.rot90(u, k=k)
    
    uniform_random = tf.random.uniform([], 0, 1.0)
    flip_cond = tf.less(uniform_random, .3)
    x = tf.cond(flip_cond, lambda: tf.image.flip_up_down(x), lambda: x)
    y = tf.cond(flip_cond, lambda: tf.image.flip_up_down(y), lambda: y)
    z = tf.cond(flip_cond, lambda: tf.image.flip_up_down(z), lambda: z)
    
    uniform_random = tf.random.uniform([], 0, 1.0)
    flip_cond = tf.less(uniform_random, .3)
    x = tf.cond(flip_cond, lambda: tf.image.flip_left_right(x), lambda: x)
    y = tf.cond(flip_cond, lambda: tf.image.flip_left_right(y), lambda: y)
    z = tf.cond(flip_cond, lambda: tf.image.flip_left_right(z), lambda: z)
    
    uniform_random = tf.random.uniform([], 0, 1.0)
    tranp_cond = tf.less(uniform_random, .3)
    x = tf.cond(tranp_cond, lambda: tf.image.transpose(x), lambda: x)
    y = tf.cond(tranp_cond, lambda: tf.image.transpose(y), lambda: y)
    z = tf.cond(tranp_cond, lambda: tf.image.transpose(z), lambda: z)
    
    #shape = tf.random.uniform(shape=[], minval=800, maxval=2000, dtype=tf.int32)
    #rescale_cond = tf.less(shape, 1472)
    #x = tf.cond(rescale_cond, lambda: tf.image.resize(x,(shape,shape)), lambda: tf.cast(x,tf.float32))
    #y = tf.cond(rescale_cond, lambda: tf.image.resize(y,(shape,shape),method = "nearest"), lambda: y)
    #z = tf.cond(rescale_cond, lambda: tf.image.resize(z,(shape,shape),method = "nearest"), lambda: z)
    
    
    #x = tf.cond(rescale_cond, lambda: tf.image.pad_to_bounding_box(x, 0, 0, 1472, 1472), lambda: x)
    #y = tf.cond(rescale_cond, lambda: tf.image.pad_to_bounding_box(y, 0, 0, 1472, 1472), lambda: y)
    #z = tf.cond(rescale_cond, lambda: tf.image.pad_to_bounding_box(z, 0, 0, 1472, 1472), lambda: z)
    
    x,y,z = normalize(x,y,z)
    
    out2 = tf.math.maximum(y,z)
    out = tf.concat([y, z,out2], -1)
    
    return x,out

  
def parse_image_tf(x):
    y = tf.strings.regex_replace(x,"images", "veins")
    z = tf.strings.regex_replace(x,"images", "artery")
    x = tf.io.read_file(x)
    x = tf.io.decode_image(x,channels = 3,expand_animations=False)
    
    y = tf.io.read_file(y)
    y = tf.io.decode_image(y, channels = 1,expand_animations=False)
    
    z = tf.io.read_file(z)
    z = tf.io.decode_image(z, channels = 1,expand_animations=False)
    
    x = tf.image.resize(x,(1444,1444))
    y = tf.image.resize(y,(1444,1444),method = "nearest")
    z = tf.image.resize(z,(1444,1444),method = "nearest")
    
    x,y,z = padding(x,y,z)
    
    x_orig = x
    mask = tf.cast(x_orig[:,:,:1]>0,tf.float32)
    # color jitter
    brightness_jitter_value = tf.random.uniform(shape=[], minval=brightness_jitter[0], maxval=brightness_jitter[1])
    contrast_jitter_value = tf.random.uniform(shape=[], minval=contrast_jitter[0], maxval=contrast_jitter[1])
    saturation_jitter_value = tf.random.uniform(shape=[], minval=saturation_jitter[0], maxval=saturation_jitter[1])
    # Define a random variable that determines whether or not to apply the jitter
    apply_jitter =  tf.less(tf.random.uniform(shape=[], minval=0, maxval=1),jitter_probability)
    x = tf.cond(apply_jitter, lambda: apply_jitter_ops(x,brightness_jitter_value,contrast_jitter_value,saturation_jitter_value,hue_jitter_value), lambda: x)
    x = tf.convert_to_tensor(x) *mask
    y = tf.cast(y, tf.float32) * mask
    z = tf.cast(z, tf.float32) * mask

    
    angle = tf.random.uniform(shape=[], minval=0, maxval=360, dtype=tf.float32)
    x = tfa.image.rotate(images = x,angles = angle * math.pi / 180)
    y = tfa.image.rotate(images = y,angles = angle * math.pi / 180)
    z = tfa.image.rotate(images = z,angles = angle * math.pi / 180)
    #x = tf.image.rot90(x, k=k)
    #y = tf.image.rot90(y, k=k)
    #z = tf.image.rot90(z, k=k)
    #u = tf.image.rot90(u, k=k)
    
    uniform_random = tf.random.uniform([], 0, 1.0)
    flip_cond = tf.less(uniform_random, .3)
    x = tf.cond(flip_cond, lambda: tf.image.flip_up_down(x), lambda: x)
    y = tf.cond(flip_cond, lambda: tf.image.flip_up_down(y), lambda: y)
    z = tf.cond(flip_cond, lambda: tf.image.flip_up_down(z), lambda: z)
    
    uniform_random = tf.random.uniform([], 0, 1.0)
    flip_cond = tf.less(uniform_random, .3)
    x = tf.cond(flip_cond, lambda: tf.image.flip_left_right(x), lambda: x)
    y = tf.cond(flip_cond, lambda: tf.image.flip_left_right(y), lambda: y)
    z = tf.cond(flip_cond, lambda: tf.image.flip_left_right(z), lambda: z)
    
    uniform_random = tf.random.uniform([], 0, 1.0)
    tranp_cond = tf.less(uniform_random, .3)
    x = tf.cond(tranp_cond, lambda: tf.image.transpose(x), lambda: x)
    y = tf.cond(tranp_cond, lambda: tf.image.transpose(y), lambda: y)
    z = tf.cond(tranp_cond, lambda: tf.image.transpose(z), lambda: z)
    
    shape = tf.random.uniform(shape=[], minval=800, maxval=2000, dtype=tf.int32)
    rescale_cond = tf.less(shape, 1472)
    x = tf.cond(rescale_cond, lambda: tf.image.resize(x,(shape,shape)), lambda: tf.cast(x,tf.float32))
    y = tf.cond(rescale_cond, lambda: tf.image.resize(y,(shape,shape),method = "nearest"), lambda: y)
    z = tf.cond(rescale_cond, lambda: tf.image.resize(z,(shape,shape),method = "nearest"), lambda: z)
    
    
    x = tf.cond(rescale_cond, lambda: tf.image.pad_to_bounding_box(x, 0, 0, 1472, 1472), lambda: x)
    y = tf.cond(rescale_cond, lambda: tf.image.pad_to_bounding_box(y, 0, 0, 1472, 1472), lambda: y)
    z = tf.cond(rescale_cond, lambda: tf.image.pad_to_bounding_box(z, 0, 0, 1472, 1472), lambda: z)
    
    x,y,z = normalize(x,y,z)
    
    out2 = tf.math.maximum(y,z)
    out = tf.concat([y, z,out2], -1)
    
    return x,out

def parse_image_tf_test(x):
    y = tf.strings.regex_replace(x,"images", "veins")
    z = tf.strings.regex_replace(x,"images", "artery")
    
    x = tf.io.read_file(x)
    x = tf.io.decode_image(x, channels = 3,expand_animations=False)
    
    y = tf.io.read_file(y)
    y = tf.io.decode_image(y, channels = 1,expand_animations=False)
    
    z = tf.io.read_file(z)
    z = tf.io.decode_image(z, channels = 1,expand_animations=False)
    
    x = tf.image.resize(x,(1444,1444))
    y = tf.image.resize(y,(1444,1444),method = "nearest")
    z = tf.image.resize(z,(1444,1444),method = "nearest")
    
    x,y,z = padding(x,y,z)
    
    x,y,z = normalize(x,y,z)
    
    out2 = tf.math.maximum(y,z)
    out = tf.concat([y, z,out2], -1)
    
    return x,out
    
def parse_image_tf_test_with_name(x):
    name = x
    y = tf.strings.regex_replace(x,"images", "veins")
    z = tf.strings.regex_replace(x,"images", "artery")
    
    x = tf.io.read_file(x)
    x = tf.io.decode_image(x, channels = 3,expand_animations=False)
    
    y = tf.io.read_file(y)
    y = tf.io.decode_image(y, channels = 1,expand_animations=False)
    
    z = tf.io.read_file(z)
    z = tf.io.decode_image(z, channels = 1,expand_animations=False)
    
    original_size = tf.shape(x)[:2]

    
    x = tf.image.resize(x,(1444,1444))
    y = tf.image.resize(y,(1444,1444),method = "nearest")
    z = tf.image.resize(z,(1444,1444),method = "nearest")
    
    x,y,z = padding(x,y,z)
    
    x,y,z = normalize(x,y,z)
    
    out2 = tf.math.maximum(y,z)
    out = tf.concat([y, z,out2], -1)
    
    return x,out,name, original_size
    
def parse_image_tf_test_with_name_without_output(x):
    name = x
    
    x = tf.io.read_file(x)
    x = tf.io.decode_image(x, channels = 3,expand_animations=False)
    
   
    x = tf.image.resize(x,(1444,1444))
   
    x,_,_ = padding(x,x,x)
    
    x,_,_ = normalize(x,x,x)
   
    
    return x,name