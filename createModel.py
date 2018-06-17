import tensorflow as tf
import os
from random import randint
import numpy as np
import traceback
import sys
from PIL import Image

# The following code is run whenever this python script is instantiated
# i.e either run or imported
# These following variables have no function block and are thus global variables

# The list of folders to be trained
folders = [ "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
			 "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
			 "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

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

# The tendorflow graph components
x = tf.placeholder(tf.float32, shape=[None, 784])
W = tf.Variable(tf.truncated_normal([784, TOTAL_ELEMENTS]), dtype=tf.float32, name="weights_0")
b = tf.Variable(tf.truncated_normal([TOTAL_ELEMENTS]), dtype=tf.float32, name="bias_0")
y = tf.nn.softmax(tf.matmul(x, W) + b)

# Training Parameters
trainingRate = 0.0001
trainingLoops = 50
batchSize = 64

# Tensorflow configuration to use CPU instead of GPU
tf_config = tf.ConfigProto(
		device_count = {'GPU': 0}
		)

# Matches each charachter to image folder name for training
def getNumber(alphabet):
	for i in range(TOTAL_ELEMENTS):
		if(alphabet == folders[i]):
			return np.eye(TOTAL_ELEMENTS, dtype=np.float32)[i]

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

def get_and_preprocess(path):
    	im = Image.open(path)
	im_g = im.convert('L')
	im_g_a = np.asarray(im_g)
	mask = im_g_a < 255
	coords = np.argwhere(mask)
	x0,y0 = coords.min(axis=0)
	x1,y1 = coords.max(axis=0)
	if( (x1-x0 > (y1-y0)) ):
		im_crop = im.crop((x0,x0,x1,x1))
	else:
		im_crop = im.crop((y0,y0,y1,y1))
	
	tf_image = tf.convert_to_tensor(np.asarray(im_crop), dtype=tf.uint8)
	return tf.image.rgb_to_grayscale(tf_image)

# This function returns the batch of images to be trained at each step
def getBatchOfLetterImages(batchSize=64):
	global startIndexOfBatch
	global imagesPathArray
	global imagesLabelsArray
	global tf_config
	
	dataset = np.ndarray(shape=(0, 784), dtype=np.float32)
	labels = np.ndarray(shape=(0, TOTAL_ELEMENTS), dtype=np.float32)
	with tf.Session(config=tf_config) as sess:
		i = startIndexOfBatch
		if(i >= len(imagesPathArray)-1):
			startIndexOfBatch = 0
			i = 0
		# for i in range(startIndexOfBatch, len(imagesPathArray)):
		while True:
			pathToImage = imagesLabelsArray[i]+imagesPathArray[i]
			lastIndexOfSlash = pathToImage.rfind("/")
			folder = pathToImage[lastIndexOfSlash - 1] 
			if(not pathToImage.endswith(".DS_Store")):
				try:
					# imageContents = tf.read_file(str(pathToImage))
					# image = tf.image.decode_png(imageContents, dtype=tf.uint8, channels=1)
					image = get_and_preprocess(str(pathToImage))
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
				except Exception as e:
					print("Unexpected Image, it's okay, skipping ({})".format(str(pathToImage)))
					print(e)
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
	yTrained = tf.placeholder(tf.float32, [None, TOTAL_ELEMENTS])

	# This is the error function
	# crossEntropy = -tf.reduce_sum(yTrained * tf.log(y))
	crossEntropy = -tf.reduce_sum(yTrained * tf.log(y+1e-20))
	# crossEntropy = -tf.reduce_sum( yTrained * tf.log( tf.clip_by_value(y, 1e-10, 1.0) ))

	# This variable represents each training step
	trainStep = tf.train.GradientDescentOptimizer(trainingRate).minimize(crossEntropy)

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
			_, error = session.run([trainStep, crossEntropy] , feed_dict={x: batchX, yTrained: batchY})
			print("Error: ", error)
		
		savedPath = saver.save(session, "./Model/model.ckpt")
		print("Model saved at: " ,savedPath)

if __name__ == '__main__':
	# In python each .py file can be imported as library or run independently
	# If run independently it comes here in __main__
	# Therefore any following code will be executed
	BeginTraining()
