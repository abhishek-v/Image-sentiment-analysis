'''
Create dictionaries for all three types of metrics. Use image name as key and sentiment value (0 for negative and 1 for positive) as values
'''

import numpy as np
import tensorflow as tf

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
    test_set.append(res)
    test_file_names.append(image)

test_set = np.array(test_set)
test_set = test_set/255

#define model
in_width, in_height, in_channels = 224, 224, 3
pretrained_resnet = tf.keras.applications.ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(in_width, in_height, in_channels),
)

model = tf.keras.models.Sequential(
    [
        pretrained_resnet,
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(3, activation="softmax")
    ]
)
model.load_weights("weights_best.hdf5")

three_agree_count = four_agree_count = five_agree_count = 0
#predict confidence scores for images
count = 0
for image in test_set:
    print("Image number:",count)
    op = -1
    conf_scores = model.predict(image)
    print(conf_scores)
    #(index: negative = 0, neutral = 1, positive = 2)
    #NEED TO TEST THIS
    if(conf_scores[0] > conf_scores[2]):
        op = 0
    else:
        op = 1

    three_agree_actual_op = three[image]
    four_agree_actual_op = four[image]
    five_agree_actual_op = five[image]

    if(three_agree_actual_op == op):
        three_agree_count = three_agree_count + 1

    if(four_agree_actual_op == op):
        four_agree_count = four_agree_count + 1

    if(five_agree_actual_op == op):
        five_agree_count = five_agree_count + 1

    count = count + 1

print("Three agree percentage:")
print(three_agree_count/three_agree)
