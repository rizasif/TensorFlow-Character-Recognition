import tensorflow as tf
import os
from random import randint
import numpy as np
import traceback
import sys

folders = [ "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
			 "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
			 "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
root = "."
datasetFolder = "/Data/training"

TOTAL_ELEMENTS = len(folders) #36

def getNumber(alphabet):
	for i in range(TOTAL_ELEMENTS):
		if(alphabet == folders[i]):
			return np.eye(TOTAL_ELEMENTS, dtype=np.float32)[i]

	# This should happen if the alphabet doesnt matches list
	assert(False)

	# For Reference
	# if(alphabet == "0"):
	# 	return np.eye(TOTAL_ELEMENTS, dtype=np.float32)[0]

	
def getListOfImages():
	global folders
	global root
	global datasetFolder
	allImagesArray = np.array([], dtype=np.str)
	allImagesLabelsArray = np.array([], dtype=np.str)
	for folder in folders:
		print("Loading Image Name of ", folder)
		currentAlphabetFolder = root+datasetFolder+"/"+folder+"/"
		imagesName = os.listdir(currentAlphabetFolder)
		allImagesArray = np.append(allImagesArray, imagesName)
		for i in range(0, len(imagesName)):
			if(i % 500 == 0):
				print("progress -> ", i)
			allImagesLabelsArray = np.append(allImagesLabelsArray, currentAlphabetFolder)
	return allImagesArray, allImagesLabelsArray

def shuffleImagesPath(imagesPathArray, imagesLabelsArray):
	print("Size of imagesPathArray is: ", len(imagesPathArray))
	for i in range(0, 100000):
		if(i % 1000 == 0):
			print("Shuflling in Progress -> ", i)
		randomIndex1 = randint(0, len(imagesPathArray)-1)
		randomIndex2 = randint(0, len(imagesPathArray)-1)
		imagesPathArray[randomIndex1], imagesPathArray[randomIndex2] = imagesPathArray[randomIndex2], imagesPathArray[randomIndex1]
		imagesLabelsArray[randomIndex1], imagesLabelsArray[randomIndex2] = imagesLabelsArray[randomIndex2], imagesLabelsArray[randomIndex1]
	return imagesPathArray, imagesLabelsArray

	
def getBatchOfLetterImages(batchSize=64):
	global startIndexOfBatch
	global imagesPathArray
	global imagesLabelsArray
	global tf_config
	
	dataset = np.ndarray(shape=(0, 784), dtype=np.float32)
	labels = np.ndarray(shape=(0, TOTAL_ELEMENTS), dtype=np.float32)
	with tf.Session(config=tf_config) as sess:
		i = startIndexOfBatch
		# for i in range(startIndexOfBatch, len(imagesPathArray)):
		while True:
			pathToImage = imagesLabelsArray[i]+imagesPathArray[i]
			lastIndexOfSlash = pathToImage.rfind("/")
			folder = pathToImage[lastIndexOfSlash - 1] 
			if(not pathToImage.endswith(".DS_Store")):
				try:
					imageContents = tf.read_file(str(pathToImage))
					image = tf.image.decode_png(imageContents, dtype=tf.uint8, channels=1)
					resized_image = tf.image.resize_images(image, [28, 28]) 
					imarray = resized_image.eval()
					imarray = imarray.reshape(784)
					appendingImageArray = np.array([imarray], dtype=np.float32)
					appendingNumberLabel = np.array([getNumber(folder)], dtype=np.float32)
					labels = np.append(labels, appendingNumberLabel, axis=0)
					dataset = np.append(dataset, appendingImageArray, axis=0)
					if(len(labels) >= batchSize):
						startIndexOfBatch = i+1
						return labels, dataset
					elif(i >= len(imagesPathArray)-1):
						startIndexOfBatch = 0
						i = 0
					# else:
					# 	print("ERROR: Mismatch Batch-Label Sizes {}-{}".format(batchSize, len(labels)))
				except:
					print("Unexpected Image, it's okay, skipping ({})".format(str(pathToImage)))
			i+=1
					
startIndexOfBatch = 0
imagesPathArray = None
imagesLabelsArray = []


tf.reset_default_graph()

x = tf.placeholder(tf.float32, shape=[None, 784])
W = tf.Variable(tf.truncated_normal([784, TOTAL_ELEMENTS]), dtype=tf.float32, name="weights_0")
b = tf.Variable(tf.truncated_normal([TOTAL_ELEMENTS]), dtype=tf.float32, name="bias_0")
y = tf.nn.softmax(tf.matmul(x, W) + b)

trainingRate = 0.005
trainingLoops = 2000
batchSize = 64

tf_config = tf.ConfigProto(
		device_count = {'GPU': 0}
		)

def BeginTraining():
	global tf_config
	global imagesPathArray
	global imagesLabelsArray

	imagesPathArray, imagesLabelsArray = getListOfImages()
	imagesPathArray, imagesLabelsArray = shuffleImagesPath(imagesPathArray, imagesLabelsArray)
	
	yTrained = tf.placeholder(tf.float32, [None, TOTAL_ELEMENTS])

	crossEntropy = -tf.reduce_sum(yTrained * tf.log(y))

	trainStep = tf.train.GradientDescentOptimizer(trainingRate).minimize(crossEntropy)

	saver = tf.train.Saver()

	with tf.Session(config=tf_config) as session:
		session.run(tf.global_variables_initializer())
		for i in range(0, trainingLoops):
			print("Training Loop number: {} of {}".format(i, trainingLoops))
			batchY, batchX = getBatchOfLetterImages(batchSize)
			print(batchX.shape, batchY.shape)
			session.run(trainStep, feed_dict={x: batchX, yTrained: batchY})
		
		savedPath = saver.save(session, "./Model/model.ckpt")
		print("Model saved at: " ,savedPath)

if __name__ == '__main__':
	BeginTraining()