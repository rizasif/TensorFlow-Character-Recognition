import tensorflow as tf
import os
from random import randint
import numpy as np
import traceback
import sys

# The following code is run whenever this python script is instantiated
# i.e either run or imported
# These following variables have no function block and are thus global variables

# The list of folders to be trained
folders = [ "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

# A string to represent working directory. "." means the current directory/folder.
root = "."

# The folder to get images to train from
datasetFolder = "/Data/training"

# Total number of elements or charachters (A-Z, 0-9), which will equal to 36
TOTAL_ELEMENTS = len(folders) #36

# Size of the reduced image
IMAGE_SIZE = 28

startIndexOfBatch = 0
imagesPathArray = None
imagesLabelsArray = []

# Initializes the tensorflow graph
tf.reset_default_graph()

# # The tendorflow graph components
# x = tf.placeholder(tf.float32, shape=[None, 784])
# W = tf.Variable(tf.truncated_normal([784, TOTAL_ELEMENTS]), dtype=tf.float32, name="weights_0")
# b = tf.Variable(tf.truncated_normal([TOTAL_ELEMENTS]), dtype=tf.float32, name="bias_0")
# y = tf.nn.softmax(tf.matmul(x, W) + b)

### set all variables

# number of neurons in each layer
input_num_units = IMAGE_SIZE*IMAGE_SIZE
hidden_num_units = int(input_num_units/2)
hidden_num_units2 = int(hidden_num_units/2)
# hidden_num_units2 = IMAGE_SIZE + 0 #784
output_num_units = TOTAL_ELEMENTS

# define placeholders
x = tf.placeholder(tf.float32, [None, input_num_units])
y = tf.placeholder(tf.float32, [None, output_num_units])

### define weights and biases of the neural network (refer this article if you don't understand the terminologies)

weights = {
    'hidden': tf.Variable(tf.random_normal([input_num_units, hidden_num_units])),
	'hidden2': tf.Variable(tf.random_normal([hidden_num_units, hidden_num_units2])),
    'output': tf.Variable(tf.random_normal([hidden_num_units2, output_num_units]))
}

biases = {
    'hidden': tf.Variable(tf.random_normal([hidden_num_units])),
	'hidden2': tf.Variable(tf.random_normal([hidden_num_units2])),
    'output': tf.Variable(tf.random_normal([output_num_units]))
}

hidden_layer = tf.add(tf.matmul(x, weights['hidden']), biases['hidden'])
# hidden_layer = tf.nn.relu(hidden_layer)
hidden_layer = tf.nn.leaky_relu(hidden_layer)

hidden_layer2 = tf.add(tf.matmul(hidden_layer, weights['hidden2']), biases['hidden2'])
hidden_layer2 = tf.nn.leaky_relu(hidden_layer2)

output_layer = tf.matmul(hidden_layer2, weights['output']) + biases['output']

# Training Parameters
trainingRate = 0.0001
trainingLoops = 50
batchSize = 8

# Tensorflow configuration to use CPU instead of GPU
tf_config = tf.ConfigProto(
		device_count = {'GPU': 0}
		)

# Matches each charachter to image folder name for training
def getNumber(alphabet):
	for i in range(TOTAL_ELEMENTS):
		if(alphabet == folders[i]):
			number = np.eye(TOTAL_ELEMENTS, dtype=np.float32)[i]
			# print("Number for {} is : {}".format(alphabet, number))
			return number

	# This should happen if the alphabet doesnt matches list
	assert(False)

# For loading training images
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

# Shuffling images for training
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


# This function returns the batch of images to be trained at each step
def getBatchOfLetterImages(batchSize=64):
	global startIndexOfBatch
	global imagesPathArray
	global imagesLabelsArray
	global tf_config
	global TOTAL_ELEMENTS
	
	dataset = np.ndarray(shape=(0, 784), dtype=np.float32)
	labels = np.ndarray(shape=(0, TOTAL_ELEMENTS), dtype=np.float32)
	with tf.Session(config=tf_config) as sess:
		i = startIndexOfBatch
		if(i >= len(imagesPathArray)-1):
			startIndexOfBatch = 0
			i = 0
		while True:
			pathToImage = imagesLabelsArray[i]+imagesPathArray[i]
			lastIndexOfSlash = pathToImage.rfind("/")
			folder = pathToImage[lastIndexOfSlash - 1] 
			if(not pathToImage.endswith(".DS_Store")):
				try:
					imageContents = tf.read_file(str(pathToImage))
					image = tf.image.decode_png(imageContents, dtype=tf.uint8, channels=1)
					resized_image = tf.image.resize_images(image, [IMAGE_SIZE, IMAGE_SIZE]) 
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

# This is the function to begin the training process
def BeginTraining():
	# In order to reference global variables, 'global' keyword is used
	global tf_config
	global imagesPathArray
	global imagesLabelsArray

	# The following reads and shuffles the training images
	imagesPathArray, imagesLabelsArray = getListOfImages()
	imagesPathArray, imagesLabelsArray = shuffleImagesPath(imagesPathArray, imagesLabelsArray)
	
	# Here we specify the neural net output
	# yTrained = tf.placeholder(tf.float32, [None, TOTAL_ELEMENTS])

	# This is the error function
	# crossEntropy = -tf.reduce_sum( yTrained * tf.log(y + (1e-50) ))
	# crossEntropy = -tf.reduce_sum( yTrained * tf.log( tf.clip_by_value(y, 1e-10, 1.0) ))
	crossEntropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output_layer, labels=y))

	# This variable represents each training step
	# trainStep = tf.train.GradientDescentOptimizer(trainingRate).minimize(crossEntropy)
	trainStep = tf.train.AdamOptimizer(trainingRate).minimize(crossEntropy)

	# This object saves the model when training is completed
	saver = tf.train.Saver()

	# Here we create a tensorflow session and start training
	with tf.Session(config=tf_config) as session:
		session.run(tf.global_variables_initializer())
		
		# Here the saver is loading the checkpoint
		if os.path.isfile("./Model/checkpoint"):
			print("Restoring Model")
			saver.restore(session, "./Model/model.ckpt")
		
		for i in range(0, trainingLoops):
			print("Training Loop number: {} of {}".format(i, trainingLoops))
			batchY, batchX = getBatchOfLetterImages(batchSize)
			print(batchX.shape, batchY.shape)
			# session.run(trainStep, feed_dict={x: batchX, yTrained: batchY})
			_, loss_val = session.run([trainStep, crossEntropy], feed_dict={x: batchX, y: batchY})
			print("Loss = {}".format(loss_val))
		
		savedPath = saver.save(session, "./Model/model.ckpt")
		print("Model saved at: " ,savedPath)

if __name__ == '__main__':
	# In python each .py file can be imported as library or run independently
	# If run independently it comes here in __main__
	# Therefore any following code will be executed
	BeginTraining()
