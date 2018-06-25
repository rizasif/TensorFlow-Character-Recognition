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
folders = [ "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
			#  "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
			#  "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

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

# Create the model
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, TOTAL_ELEMENTS])
W = tf.Variable(tf.truncated_normal([784, TOTAL_ELEMENTS]))
b = tf.Variable(tf.truncated_normal([TOTAL_ELEMENTS]))
y = tf.nn.softmax(tf.matmul(x, W) + b)

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)


def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1,IMAGE_SIZE,IMAGE_SIZE,1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)


W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#--------------
W_conv3 = weight_variable([5, 5, 64, 128])
b_conv3 = bias_variable([128])

h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
h_pool3 = max_pool_2x2(h_conv3)

W_conv4 = weight_variable([5, 5, 128, 256])
b_conv4 = bias_variable([256])

h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)
h_pool4 = max_pool_2x2(h_conv4)

W_conv5 = weight_variable([5, 5, 256, 512])
b_conv5 = bias_variable([512])

h_conv5 = tf.nn.relu(conv2d(h_pool4, W_conv5) + b_conv5)
h_pool5 = max_pool_2x2(h_conv5)
#--------------

W_fc1 = weight_variable([512, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool5, [-1, 512])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, TOTAL_ELEMENTS])
b_fc2 = bias_variable([TOTAL_ELEMENTS])

y_conv=tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2)
# y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# Training Parameters
trainingRate = 0.0001
trainingLoops = 500
batchSize = 32

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

# def get_and_preprocess(path):
#     	im = Image.open(path)
# 	im_g = im.convert('L')
# 	im_g_a = np.asarray(im_g)
# 	mask = im_g_a < 255
# 	coords = np.argwhere(mask)
# 	x0,y0 = coords.min(axis=0)
# 	x1,y1 = coords.max(axis=0)
# 	if( (x1-x0 > (y1-y0)) ):
# 		im_crop = im.crop((x0,x0,x1,x1))
# 	else:
# 		im_crop = im.crop((y0,y0,y1,y1))
	
# 	tf_image = tf.convert_to_tensor(np.asarray(im_crop), dtype=tf.uint8)
# 	return tf.image.rgb_to_grayscale(tf_image)

# This function returns the batch of images to be trained at each step
def getBatchOfLetterImages(batchSize=64):
	global startIndexOfBatch
	global imagesPathArray
	global imagesLabelsArray
	global tf_config
	
	dataset = np.ndarray(shape=(0, 784), dtype=np.float32)
	labels = np.ndarray(shape=(0, TOTAL_ELEMENTS), dtype=np.float32)
	with tf.Session(config=tf_config) as sess:
		i = startIndexOfBatch + randint(0, 5)
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
					# print(str(pathToImage))
					imageContents = tf.read_file(str(pathToImage))
					image = tf.image.decode_png(imageContents, dtype=tf.uint8, channels=1)
					# image = get_and_preprocess(str(pathToImage))
					resized_image = tf.image.resize_images(image, [IMAGE_SIZE, IMAGE_SIZE]) 
					imarray = resized_image.eval()
					imarray = imarray.reshape(784)
					appendingImageArray = np.array([imarray], dtype=np.float32)
					appendingNumberLabel = np.array([getNumber(folder)], dtype=np.float32)
					# print(appendingNumberLabel)
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
	
	# Define loss and optimizer
	# cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
	cross_entropy = -tf.reduce_sum(y_*tf.log( tf.clip_by_value(y_conv, 1e-10, 1.0) ))
	regularizer = tf.nn.l2_loss(W_conv1)+tf.nn.l2_loss(W_conv2)
	loss = tf.reduce_mean(cross_entropy+0.0001*regularizer)

	# train_step = tf.train.AdamOptimizer(trainingRate).minimize(cross_entropy)
	train_step = tf.train.AdamOptimizer(trainingRate).minimize(loss)
	
	correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

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
			# print(batchY)
			if i%10 == 0:
				train_accuracy = accuracy.eval(feed_dict={x:batchX, y_: batchY, keep_prob: 1.0})
				print("step %d, training accuracy %g"%(i, train_accuracy))
				
			train_step.run(feed_dict={x: batchX, y_: batchY, keep_prob: 1.0})
		
		savedPath = saver.save(session, "./Model/model.ckpt")
		print("Model saved at: " ,savedPath)

if __name__ == '__main__':
	# In python each .py file can be imported as library or run independently
	# If run independently it comes here in __main__
	# Therefore any following code will be executed
	BeginTraining()
