import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.models import load_model
import numpy as np
import os
from scipy.misc import imread, imresize
import sys

folders = [ "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
	
# The folder to get images to train from
datasetFolder = "./Data/training"
modelfile = "Model/cnn.h5"

# Total number of elements or charachters (A-Z, 0-9), which will equal to 36
TOTAL_ELEMENTS = len(folders) #36

# Size of the reduced image
IMAGE_SIZE = 28

batch_size = 128
num_classes = len(folders)
epochs = 100

# Matches each charachter to image folder name for training
def getNumber(alphabet):
	for i in range(TOTAL_ELEMENTS):
		if(alphabet == folders[i]):
			return np.eye(TOTAL_ELEMENTS, dtype=np.float32)[i]

	# This should happen if the alphabet doesnt matches list
	sys.exit("Number not found")

def getListOfImages():
	allImagesArray = np.array([], dtype=np.str)
	allImagesLabelsArray = np.array([], dtype=np.str)
	for folder in folders:
		print("Loading Image Name of ", folder)
		currentAlphabetFolder = datasetFolder+"/"+folder+"/"
		imagesName = os.listdir(currentAlphabetFolder)
		allImagesArray = np.append(allImagesArray, imagesName)
		for i in range(0, len(imagesName)):
			if(i % 500 == 0):
				print("progress -> {} of {}".format(i, len(imagesName)))
			allImagesLabelsArray = np.append(allImagesLabelsArray, currentAlphabetFolder)
	return allImagesArray, allImagesLabelsArray

def shuffleImagesPath(imagesPathArray, imagesLabelsArray):
	print("Size of imagesPathArray is: ", len(imagesPathArray))
	print("Shuffling in progress")
	orig_paths = np.array(imagesPathArray)
	orig_labels = np.array(imagesLabelsArray)
	permutations = np.random.permutation(orig_labels.shape[0])
	imagesPathArray = orig_paths[permutations]
	imagesLabelsArray =  orig_labels[permutations]
	return imagesPathArray, imagesLabelsArray

def getBatchOfLetterImages(imagesPathArray, imagesLabelsArray):
	dataset = np.ndarray(shape=(0, 784), dtype=np.float32)
	labels = np.ndarray(shape=(0, TOTAL_ELEMENTS), dtype=np.float32)
	for i in range(len(imagesPathArray)):
		pathToImage = imagesLabelsArray[i]+imagesPathArray[i]
		lastIndexOfSlash = pathToImage.rfind("/")
		folder = pathToImage[lastIndexOfSlash - 1] 
		if(not pathToImage.endswith(".DS_Store")):
			# print(str(pathToImage))
			image = imread(pathToImage,mode='L')
			resized_image = imresize(image,(IMAGE_SIZE,IMAGE_SIZE))
			imarray = np.array(resized_image)
			imarray = imarray.reshape(IMAGE_SIZE*IMAGE_SIZE)
			appendingImageArray = np.array([imarray], dtype=np.float32)
			appendingNumberLabel = np.array([getNumber(folder)], dtype=np.float32)
			# print(appendingNumberLabel)
			labels = np.append(labels, appendingNumberLabel, axis=0)
			dataset = np.append(dataset, appendingImageArray, axis=0)
	return dataset, labels

# This is the function to begin the training process
def BeginTraining():
	#Dataset
	paths, labels = getListOfImages()
	paths, labels = shuffleImagesPath(paths, labels)
	images, labels = getBatchOfLetterImages(paths, labels)

	train_split = int(0.8*len(labels))
	test_split = len(labels) - train_split


	x_train = images[0:train_split]
	y_train = labels[0:train_split]

	x_test = images[train_split:len(labels)]
	y_test = labels[train_split:len(labels)]

	print("train: {} test: {}".format(len(x_test), len(y_test)))

	x_train = x_train.reshape(x_train.shape[0], IMAGE_SIZE, IMAGE_SIZE, 1)
	x_test = x_test.reshape(x_test.shape[0], IMAGE_SIZE, IMAGE_SIZE, 1)
	input_shape = (IMAGE_SIZE, IMAGE_SIZE, 1)

	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')
	x_train /= 255
	x_test /= 255
	print('x_train shape:', x_train.shape)
	print(x_train.shape[0], ' train samples')
	print(x_test.shape[0], ' test samples')


	if(os.path.isfile(modelfile)):
		model = load_model(modelfile)
		print("Loading Saved Model")
	else:
		print("Saved Model Not Found")
		model = Sequential()
		model.add(Conv2D(32, kernel_size=(3, 3),
						activation='relu',
						input_shape=input_shape))
		model.add(Conv2D(64, (3, 3), activation='relu'))
		model.add(Conv2D(128, (3, 3), activation='relu'))
		model.add(Conv2D(256, (3, 3), activation='relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))
		model.add(Flatten())
		model.add(Dense(512, activation='relu'))
		model.add(Dropout(0.5))
		model.add(Dense(num_classes, activation='softmax'))

		model.compile(loss=keras.losses.categorical_crossentropy,
					optimizer=keras.optimizers.Adadelta(),
					metrics=['accuracy'])


	model.fit(x_train, y_train,
			batch_size=batch_size,
			epochs=epochs,
			verbose=1,
			validation_data=(x_test, y_test))

	model.save(modelfile)
	sys.exit("Training Complete")

if __name__ == '__main__':
	# In python each .py file can be imported as library or run independently
	# If run independently it comes here in __main__
	# Therefore any following code will be executed
	BeginTraining()
