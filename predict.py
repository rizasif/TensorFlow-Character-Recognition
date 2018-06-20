import sys
import tensorflow as tf
from PIL import Image,ImageFilter
import numpy as np
import os

# Size of reduced image
IMAGE_SIZE = 28

# For printing
folders = [ "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
            #  "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
            #  "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

# Total number of charachters to predict from
TOTAL_ELEMENTS = len(folders)

# This is the function to predict the letter
def predictLetter(imvalue):

    # Initiate the tensorflow graph
    tf.reset_default_graph()

    # Create the model
    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, TOTAL_ELEMENTS]))
    b = tf.Variable(tf.zeros([TOTAL_ELEMENTS]))

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

    x_image = tf.reshape(x, [-1,28,28,1])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)


    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([1024, TOTAL_ELEMENTS])
    b_fc2 = bias_variable([TOTAL_ELEMENTS])

    y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    
    # Initializing all components
    init_op = tf.global_variables_initializer()

    # Saver is used to load saved checkpoint
    saver = tf.train.Saver()

    # Tensorflow configuration to use CPU over GPU
    tf_config = tf.ConfigProto(
        device_count = {'GPU': 0})

    # Session to start tensorflow graph
    with tf.Session(config=tf_config) as sess:
        sess.run(init_op)

        # Here the saver is loading the checkpoint
        saver.restore(sess, "./Model/model.ckpt")
   
        # prediction=tf.argmax(y_conv,1)
        # index = prediction.eval(feed_dict={x: [imvalue]}, session=sess)
        # return folders[index[0]]
        return sess.run(y_conv, feed_dict={x:[imvalue], keep_prob: 1.0})

# This function prepares the image
def getImage(image):

    # Converting image to gray scale
    # im = image.convert('L')

    im_g = image.convert('L')
    im_g_a = np.asarray(im_g)
    mask = im_g_a < 255
    coords = np.argwhere(mask)
    x0,y0 = coords.min(axis=0)
    x1,y1 = coords.max(axis=0)
    if( (x1-x0 > (y1-y0)) ):
        im = im_g.crop((x0,x0,x1,x1))
    else:
        im = im_g.crop((y0,y0,y1,y1))

    # Getting width and height of image
    width = float(im.size[0])
    height = float(im.size[1])

    # Resizing the image to IMAGE_SIZExIMAGE_SIZE
    newImage = Image.new('L', (IMAGE_SIZE, IMAGE_SIZE), (255)) 
    
    # # Checking orientation of image and processing accordingly
    if width > height: 
        nheight = int(round((20.0/width*height),0)) 
        if (nheight == 0): 
            nheight = 1
        img = im.resize((20,nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wtop = int(round(((IMAGE_SIZE - nheight)/2),0)) 
        newImage.paste(img, (4, wtop)) 
    else:
        nwidth = int(round((20.0/height*width),0)) 
        if (nwidth == 0): 
            nwidth = 1
         
        img = im.resize((nwidth,20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wleft = int(round(((IMAGE_SIZE - nwidth)/2),0)) 
        newImage.paste(img, (wleft, 4)) 
    
    # Storing the processed image in a list (here the list has only this image)
    tv = list(newImage.getdata()) 
    
    
    tva = [ (255-x)*1.0/255.0 for x in tv] 
    return tva
    

def main(argv):
    im = Image.open(argv).convert('L')
    imvalue = getImage(im)
    predictedLetter = predictLetter(imvalue)
    print (predictedLetter)
    print (folders[ np.argmax(predictedLetter, 1)[0] ] )
    
if __name__ == "__main__":
    # Whenever the script runs independently this main function is called
    # sys.argv provides the image path when used from cmd
    main(sys.argv[1])
