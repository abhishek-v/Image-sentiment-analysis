'''
Create dictionaries for all three types of metrics. Use image name as key and sentiment value (0 for negative and 1 for positive) as values
'''

import numpy as np
import tensorflow as tf
import os
import cv2

AUTOTUNE = tf.data.experimental.AUTOTUNE
tf.enable_eager_execution()
in_width, in_height, in_channels = 224, 224, 3

five = {}
four = {}
three = {}

f = open("five_agree.txt","r")
for line in f:
    line = line.rstrip("\n")
    line = line.split(" ")
    five[line[0]] = int(line[1])

f = open("four_agree.txt","r")
for line in f:
    line = line.rstrip("\n")
    line = line.split(" ")
    four[line[0]] = int(line[1])

f = open("three_agree.txt","r")
for line in f:
    line = line.rstrip("\n")
    line = line.split(" ")
    three[line[0]] = int(line[1])

#load images
os.chdir("./Agg_AMT_Candidates")
images = os.listdir()
test_set = []
test_file_names = []
for image in images:
    img = cv2.imread(image)
    res = cv2.resize(img, dsize=(224,224), interpolation=cv2.INTER_CUBIC)
    resize_image = res.reshape([-1, 224, 224, 3])
    test_set.append(resize_image)
    test_file_names.append(image)

# print(len(test_set))
# print(len(three.keys()))
# print(len(four.keys()))
# print(len(five.keys()))
# exit(1)
test_set = np.array(test_set)
test_set = test_set/255

os.chdir("..")

#define model
in_width, in_height, in_channels = 224, 224, 3


ip = input("Which model to use? 1 for DenseNet121, 2 for VGG19: ")
if(int(ip) == 1):
    print("Using densenet")
    pretrained_resnet = tf.keras.applications.DenseNet121(
        weights="imagenet",
        include_top=False,
        input_shape=(in_width, in_height, in_channels),
    )
    model = tf.keras.models.Sequential(
        [
            pretrained_resnet,
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(3, activation="softmax",kernel_regularizer=tf.keras.regularizers.l2(l=0.1))
        ]
    )
    for layer in pretrained_resnet.layers:
        layer.trainable = False
    model.load_weights("weights_best_densenet.hdf5")
else:
    print("Using VGG19")
    pretrained_resnet = tf.keras.applications.VGG19(
        weights="imagenet",
        include_top=False,
        input_shape=(in_width, in_height, in_channels),
    )
    model = tf.keras.models.Sequential(
        [
            pretrained_resnet,
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(3, activation="softmax",kernel_regularizer=tf.keras.regularizers.l2(l=0.1))
        ]
    )
    for layer in pretrained_resnet.layers:
        layer.trainable = False
    model.load_weights("weights_best_vgg.hdf5")

three_agree_count = four_agree_count = five_agree_count = 0
#predict confidence scores for images

for i in range(len(test_set)):
    print("Image number:",i)
    op = -1
    conf_scores = model.predict(test_set[i])
    print(conf_scores)
    #(index: negative = 0, neutral = 1, positive = 2)
    #NEED TO TEST THIS
    if(conf_scores[0][0] > conf_scores[0][2]):
        op = 0
    else:
        op = 1
    print(test_file_names[i])
    three_agree_actual_op = four_agree_actual_op = five_agree_actual_op = -1
    try:
        three_agree_actual_op = three[test_file_names[i]]
    except:
        pass
    try:
        four_agree_actual_op = four[test_file_names[i]]
    except:
        pass
    try:
        five_agree_actual_op = five[test_file_names[i]]
    except:
        pass

    if(three_agree_actual_op == op):
        three_agree_count = three_agree_count + 1

    if(four_agree_actual_op == op):
        four_agree_count = four_agree_count + 1

    if(five_agree_actual_op == op):
        five_agree_count = five_agree_count + 1

print(three_agree_count)
print(four_agree_count)
print(five_agree_count)

print("Three agree percentage:")
print((three_agree_count/len(three.keys()))*100)

print("Four agree percentage:")
print((four_agree_count/len(four.keys()))*100)

print("Five agree percentage:")
print((five_agree_count/len(five.keys()))*100)
