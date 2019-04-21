import pathlib
import csv
import tensorflow as tf
import pickle
import time
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import random
import cv2
import numpy as np

AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 32
tf.enable_eager_execution()
in_width, in_height, in_channels = 224, 224, 3

def preprocess_image(image):
  image = tf.image.decode_jpeg(image, channels=3)
  image = tf.image.resize_images(image, [224, 224])
  image /= 255.0  # normalize to [0,1] range

  return image


def load_and_preprocess_image(path):
  image = tf.read_file(path)
  return preprocess_image(image)

data_root = pathlib.Path("./data")

all_image_paths = list(data_root.glob('*/*'))
all_image_paths = [str(path) for path in all_image_paths]



d = {}
with open('./t4sa_text_sentiment.tsv') as tsvfile:
  reader = csv.reader(tsvfile, delimiter='\t')
  next(reader)
  for row in reader:
    temp1 = row[0]
    temp2 = row[1:]
    try:
         temp2 = [float(i) for i in temp2]
    except:
        # print(temp2)
        exit(1)
    d[temp1] = temp2


#create validation set
all_val_image_paths = []
val_labels = []
indices = random.sample(range(1, len(all_image_paths)), 3200 )
for index in indices:
    try:
        img_name = all_image_paths.pop(index)
        # img = cv2.imread(img_name)
        # res = cv2.resize(img, dsize=(224,224), interpolation=cv2.INTER_CUBIC)
        # val_images.append(res)
        all_val_image_paths.append(img_name)
        fname = img_name.split("/")[-1]
        fname = fname.rstrip(".jpg")
        fname = fname.split("-")[0]
        temp = d[fname]
        b = [1 if x == max(temp) else 0 for x in temp]
        val_labels.append(b)
    except:
        continue

# val_images = np.array(val_images,dtype=np.uint8)
# val_labels = np.array(val_labels)

# val_images = val_images/255
# print("Number of examples in validation set: ",len(val_images))


image_count = len(all_image_paths)


pos = neu = neg = 0
all_image_labels = []
for fpath in all_image_paths:
    fname = fpath.split("/")[-1]
    fname = fname.rstrip(".jpg")
    fname = fname.split("-")[0]
    temp = d[fname]
    b = [1 if x == max(temp) else 0 for x in temp]
    all_image_labels.append(b)
    if(b[0] == 1):
        neg = neg + 1
    elif(b[1] == 1):
        neu = neu + 1
    else:
        pos = pos + 1

print("pos,neu,neg",pos,neu,neg)

# for image_path in all_image_paths:
#     print(all_image_paths)

path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int64))
image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))
print('image shape: ', image_label_ds.output_shapes[0])
print('label shape: ', image_label_ds.output_shapes[1])
print('types: ', image_label_ds.output_types)
print()
print(image_label_ds)

ds = tf.data.Dataset.from_tensor_slices((all_image_paths, all_image_labels))

# The tuples are unpacked into the positional arguments of the mapped function
def load_and_preprocess_from_path_label(path, label):
  return load_and_preprocess_image(path), label

image_label_ds = ds.map(load_and_preprocess_from_path_label)
print(image_label_ds)
ds = image_label_ds.shuffle(buffer_size=3000)
ds = ds.repeat()
ds = ds.batch(BATCH_SIZE)
# `prefetch` lets the dataset fetch batches, in the background while the model is training.
ds = ds.prefetch(buffer_size=AUTOTUNE)



'''for val image'''
path_ds_val = tf.data.Dataset.from_tensor_slices(all_val_image_paths)
image_ds_val = path_ds_val.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
label_ds_val = tf.data.Dataset.from_tensor_slices(tf.cast(val_labels, tf.int64))
image_label_ds_val = tf.data.Dataset.zip((image_ds_val, label_ds_val))
print('image shape val: ', image_label_ds_val.output_shapes[0])
print('label shape val: ', image_label_ds_val.output_shapes[1])
print('types val: ', image_label_ds_val.output_types)
print()
print(image_label_ds_val)

ds_val = tf.data.Dataset.from_tensor_slices((all_image_paths, all_image_labels))

image_label_ds_val = ds_val.map(load_and_preprocess_from_path_label)
print(image_label_ds_val)
ds_val = image_label_ds_val.shuffle(buffer_size=3000)
ds_val = ds_val.repeat()
ds_val = ds_val.batch(BATCH_SIZE)
# `prefetch` lets the dataset fetch batches, in the background while the model is training.
ds_val = ds_val.prefetch(buffer_size=AUTOTUNE)



ds_val = tf.data.Dataset.from_tensor_slices((all_val_image_paths, val_labels))

image_label_ds_val = ds_val.map(load_and_preprocess_from_path_label)
print(image_label_ds_val)
ds_val = image_label_ds_val.shuffle(buffer_size=1)
ds_val = ds_val.repeat()
ds_val = ds_val.batch(BATCH_SIZE)
# `prefetch` lets the dataset fetch batches, in the background while the model is training.
ds_val = ds_val.prefetch(buffer_size=AUTOTUNE)

'''   '''
pretrained_resnet = tf.keras.applications.ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(in_width, in_height, in_channels),
)

for layer in pretrained_resnet.layers:
    layer.trainable = False

model = tf.keras.models.Sequential(
    [
        pretrained_resnet,
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),
        #tf.keras.layers.Dense(128, activation="relu",kernel_regularizer=tf.keras.regularizers.l2(l=0.1)),
        #tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(256, activation="relu",kernel_regularizer=tf.keras.regularizers.l2(l=0.1)),
        #tf.keras.layers.Dropout(0.4),
        #tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(3, activation="softmax",kernel_regularizer=tf.keras.regularizers.l2(l=0.01))
    ]
)

filepath="weights_best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max',save_weights_only=True)
callbacks_list = [checkpoint]

image_batch, label_batch = next(iter(ds))
model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.00001),metrics=['acc','mse', 'mae'])
history = model.fit(ds, epochs=1, steps_per_epoch = 500, callbacks=callbacks_list, batch_size=32,nb_epoch=50,verbose=1,validation_data=ds_val,validation_steps=100)
op = open("stats","wb")
pickle.dump(history.history['mean_absolute_error'],op)
pickle.dump(history.history['mean_squared_error'],op)
pickle.dump(history.history['acc'],op)
op.close()
