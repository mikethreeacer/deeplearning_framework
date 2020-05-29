import glob

import numpy as np
import cv2

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array


my_train_img = "/home/jeff/Giga_Face_img/196/RGB/*/*"
# make sure your image has the right label, so no shuffle
my_train_img_dataset = tf.data.Dataset.list_files(my_train_img, shuffle=False)
# sorted() with name to match image
my_train = sorted(glob.glob(my_train_img))

i = 0
for element in my_train_img_dataset:
  print(element.numpy())
  i += 1
  if i >10:
    break
print(my_train[:11])

my_train_img_labels = [[1., 0.] if x.split('/')[-2][-4:] == 'mask' else [0., 1.] for x in my_train]
my_train_img_labels_dataset = tf.data.Dataset.from_tensor_slices(my_train_img_labels)

train_dataset = tf.data.Dataset.zip((my_train_img_dataset, my_train_img_labels_dataset))


img_size_list = [96, 128, 160, 192, 224]
pool_size_list = [3, 4, 5, 6, 7]
i = 4
img_size = img_size_list[i]
pool_size = pool_size_list[i]

shuffle_buffer_size = len(glob.glob(my_train_img))
val_rate = 0.2
batch_size = 32
INIT_LR = 1e-4
EPOCHS = 50

checkpoint_dir = '/home/jeff/test/COFW/trained_model/'
save_file_name = 'me_0196samall_96_norm'

def aug_preprocess(img_path, label):
  img = tf.io.read_file(img_path)
  img = tf.io.decode_image(img, channels=3, expand_animations = False)
  img = tf.image.resize(img, (img_size, img_size))
  #img = tf.image.convert_image_dtype(img, tf.float32)
  img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
  img = img.numpy()
  img = tf.keras.preprocessing.image.random_rotation(img, 20, row_axis=0, col_axis=1, channel_axis=2)
  #img = tf.keras.preprocessing.image.random_zoom(img, (0.1, 0.1), row_axis=0, col_axis=1, channel_axis=2)
  img = tf.keras.preprocessing.image.random_shift(img, 0.2, 0.2, row_axis=0, col_axis=1, channel_axis=2)
  img = tf.keras.preprocessing.image.random_shear(img, 0.15, row_axis=0, col_axis=1, channel_axis=2)
  img = tf.convert_to_tensor(img)
  img = tf.image.random_flip_left_right(img)
  return img, label

def noaug_preprocess(img_path, label):
  img = tf.io.read_file(img_path)
  img = tf.io.decode_image(img, channels=3, expand_animations = False)
  img = tf.image.resize(img, (img_size, img_size))
  #img = tf.image.convert_image_dtype(img, tf.float32)
  img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
  return img, label

def set_shapes(image, label):
    #image.set_shape((224, 224, 3))
    image.set_shape((None, None, None))
    label.set_shape((2,))
    return image, label


def prepare_for_training(ds, cache=True, shuffle_buffer_size=shuffle_buffer_size):
  if cache:
    if isinstance(cache, str):
      ds = ds.cache(cache)
    else:
      ds = ds.cache()
  # seed 53, epoch 50 is good!
  ds = ds.shuffle(buffer_size = shuffle_buffer_size, seed = 53 ,reshuffle_each_iteration=False)
  test_dataset = ds.take(int(val_rate*shuffle_buffer_size))
  train_dataset = ds.skip(shuffle_buffer_size - int(val_rate*shuffle_buffer_size))

  test_dataset = test_dataset.shuffle(buffer_size = int(val_rate*shuffle_buffer_size))
  train_dataset = train_dataset.shuffle(buffer_size = shuffle_buffer_size - int(val_rate*shuffle_buffer_size))

  test_dataset = test_dataset.map(noaug_preprocess, num_parallel_calls = tf.data.experimental.AUTOTUNE)
  #train_dataset = train_dataset.map(aug_preprocess, num_parallel_calls = tf.data.experimental.AUTOTUNE)
  train_dataset = train_dataset.map(lambda x, y: tf.py_function(aug_preprocess, [x, y], [tf.float32, tf.float32]), num_parallel_calls = tf.data.experimental.AUTOTUNE)
  train_dataset = train_dataset.map(set_shapes, num_parallel_calls = tf.data.experimental.AUTOTUNE)

  test_dataset = test_dataset.batch(batch_size)
  train_dataset = train_dataset.batch(batch_size)

  test_dataset = test_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
  train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

  return train_dataset, test_dataset

train_dataset, test_dataset = prepare_for_training(train_dataset)


# load the MobileNetV2 network, ensuring the head FC layer sets are
# left off
baseModel = MobileNetV2(input_shape = (img_size, img_size, 3), weights="imagenet", include_top=False)
#baseModel = MobileNetV2(weights="imagenet", include_top=False,
#	 input_tensor=Input(shape=(224, 224, 3)))
# construct the head of the model that will be placed on top of the
# the base model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(pool_size, pool_size))(headModel)
#headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)
# place the head FC model on top of the base model (this will become
# the actual model we will train)
model = Model(inputs=baseModel.input, outputs=headModel)
# loop over all layers in the base model and freeze them so they will
# *not* be updated during the first training process
for layer in baseModel.layers:
	layer.trainable = False

# compile our model
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])
# train the head of the network
print("[INFO] training head...")

#cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_dir, save_weights_only=False, save_best_only=True, verbose=1)
earlystop_cb = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', min_delta=0.001, patience=10, verbose=1, restore_best_weights=True)

H = model.fit(
	x = train_dataset,
	validation_data=test_dataset,
	epochs=EPOCHS,
  callbacks = [earlystop_cb],
  verbose = 1)

model.save('/home/jeff/test/COFW/trained_model/{}'.format(save_file_name), save_format="h5")