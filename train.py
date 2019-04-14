import pickle
import os
import tensorflow as tf
import keras
import numpy as np
import cv2
from tensorflow.keras.optimizers import Adam

f = open("avail_files.pkl","rb")
avail_files = pickle.load(f)

os.chdir("./data")

train_images = []
train_op = []
#train_images = np.empty((224,224,3), int)
files = os.listdir()
for file in files:
  os.chdir(file)
  sub_files = os.listdir()
  for sub_file in sub_files:
    name = sub_file.split("-")[0]
    #print("reading file",name)
    img = cv2.imread(sub_file)
    res = cv2.resize(img, dsize=(224,224), interpolation=cv2.INTER_CUBIC)
    #res = np.expand_dims(res, axis=0)
    train_images.append(res)
    temp = avail_files[name]
    b = [1 if x == max(temp) else 0 for x in temp]
    train_op.append(b)
    #uncomment following to display image
    # cv2.imshow('image', train_images[0])
    # cv2.waitKey(0)
  os.chdir("..")

in_width, in_height, in_channels = 224, 224, 3 #variable size input
print("done reading file")
pretrained_resnet = tf.keras.applications.ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(in_width, in_height, in_channels),
)
print("resnet initiate", name)
model = tf.keras.models.Sequential(
    [
        pretrained_resnet,
        tf.keras.layers.Flatten(),
        #tf.keras.layers.Dense(1024, activation="relu"),
        #tf.keras.layers.Dropout(0.4),
        #tf.keras.layers.Dense(1024, activation="relu"),
        tf.keras.layers.Dense(3, activation="softmax")
    ]
)
print("resnet initiated", name)
print(model.output_shape)

train_images = np.array(train_images)


train_op = np.array(train_op)
print("numpy arrays created", name)
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc','mse', 'mae'])
history = model.fit(train_images,train_op,batch_size=64,nb_epoch=15,verbose=1,validation_split=0.2,shuffle=True)

op = open("stats","w")
pickle.dump(history.history['mean_absolute_error'],op)
pickle.dump(history.history['mean_squared_error'],op)
pickle.dump(history.history['acc'],op)
op.close()
model.save('sent_model.h5')

#IMPORTANT: SAVE MODEL
