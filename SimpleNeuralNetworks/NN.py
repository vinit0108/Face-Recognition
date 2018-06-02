from sklearn.preprocessing import LabelEncoder

from sklearn.cross_validation import train_test_split
from keras.models import Sequential

from keras.layers import Activation
from keras.optimizers import SGD
from keras import optimizers
from keras.layers import Dense
from keras.utils import np_utils

from imutils import paths
import numpy as np

import argparse

import cv2


import os

def image_to_feature_vector(image, size=(32, 32)):

	return cv2.resize(image, size).flatten()

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")

args = vars(ap.parse_args())

print("describing images...")
imagePaths = list(paths.list_images(args["dataset"]))

data = []
labels = []

for (i, imagePath) in enumerate(imagePaths):
	image = cv2.imread(imagePath)

	label = imagePath.split(os.path.sep)[-1].split("_")[0]

	features = image_to_feature_vector(image)
	data.append(features)
	labels.append(label)


	if i > 0 and i % 5000 == 0:
		print("processed {}/{}".format(i, len(imagePaths)))


le = LabelEncoder()
labels = le.fit_transform(labels)

data = np.array(data) / 255.0
labels = np_utils.to_categorical(labels, 82)

print("[INFO] constructing training/testing split...")
(trainData, testData, trainLabels, testLabels) = train_test_split(
	data, labels, test_size=0.30, random_state=42)
model = Sequential()

model.add(Dense(768, input_dim=3072, init="uniform",
	activation="relu"))
model.add(Dense(384, init="uniform", activation="relu"))
model.add(Dense(82))
model.add(Activation("softmax"))


print("compiling model...")

sgd = SGD(lr=0.1)

model.compile(loss="binary_crossentropy", optimizer=sgd,
	metrics=["accuracy"])
model.fit(trainData, trainLabels, nb_epoch=50, batch_size=128,
	verbose=1)

print(" evaluating on testing set...")
(loss, accuracy) = model.evaluate(testData, testLabels,
	batch_size=1, verbose=1)
## if accuracy > threshold take testlabel testdata decompress the image and store into train folder
print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,
	accuracy * 100))
