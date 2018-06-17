import sys
import tensorflow as tf
from PIL import Image,ImageFilter
import numpy as np

# Total number of charachters to predict from
TOTAL_ELEMENTS = 10

# Size of reduced image
IMAGE_SIZE = 28

# This is the function to predict the letter
def predictLetter(imvalue):

    # Initiate the tensorflow graph
    # tf.reset_default_graph()

    # Empty tensorflow components
    # x = tf.placeholder(tf.float32, shape=[None, 784])
    # W = tf.Variable(tf.truncated_normal([784, TOTAL_ELEMENTS]), dtype=tf.float32, name="weights_0")
    # b = tf.Variable(tf.truncated_normal([TOTAL_ELEMENTS]), dtype=tf.float32, name="bias_0")
    # y = tf.nn.softmax(tf.matmul(x, W) + b)

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
   
        # prediction=tf.argmax(output_layer,1)
        # return prediction.eval(feed_dict={x: [imvalue]}, session=sess)
        return sess.run(output_layer, feed_dict={x:[imvalue]})

# This function prepares the image
def getImage(image):

    # Converting image to gray scale
    im = image.convert('L')

    # Getting width and height of image
    width = float(im.size[0])
    height = float(im.size[1])

    # Resizing the image to IMAGE_SIZExIMAGE_SIZE
    newImage = Image.new('L', (IMAGE_SIZE, IMAGE_SIZE), (255)) 
    
    
    # Storing the processed image in a list (here the list has only this image)
    tv = list(newImage.getdata()) 
    
    
    # tva = [ (255-x)*1.0/255.0 for x in tv]
    tva = tv 
    return tva

# This function reads and prepares the image, you can ignore this for now
# def imageprepare(argv):

#     im = Image.open(argv).convert('L')
#     width = float(im.size[0])
#     height = float(im.size[1])
#     newImage = Image.new('L', (IMAGE_SIZE, IMAGE_SIZE), (255)) 
    
#     if width > height: 
#         nheight = int(round((20.0/width*height),0)) 
#         if (nheight == 0): 
#             nheight = 1
#         img = im.resize((20,nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
#         wtop = int(round(((IMAGE_SIZE - nheight)/2),0)) 
#         newImage.paste(img, (4, wtop)) 
#     else:
        
#         nwidth = int(round((20.0/height*width),0)) 
#         if (nwidth == 0): 
#             nwidth = 1
         
#         img = im.resize((nwidth,20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
#         wleft = int(round(((IMAGE_SIZE - nwidth)/2),0)) 
#         newImage.paste(img, (wleft, 4)) 
    
    

#     tv = list(newImage.getdata()) 
    
    
#     tva = [ (255-x)*1.0/255.0 for x in tv] 
#     return tva

def imageprepare(argv):
    
    im = Image.open(argv).convert('L')
    width = float(im.size[0])
    height = float(im.size[1])
    newImage = Image.new('L', (IMAGE_SIZE, IMAGE_SIZE), (255)) 
    
    # if width > height: 
    #     nheight = int(round((20.0/width*height),0)) 
    #     if (nheight == 0): 
    #         nheight = 1
    #     img = im.resize((20,nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
    #     wtop = int(round(((IMAGE_SIZE - nheight)/2),0)) 
    #     newImage.paste(img, (4, wtop)) 
    # else:
        
    #     nwidth = int(round((20.0/height*width),0)) 
    #     if (nwidth == 0): 
    #         nwidth = 1
         
    #     img = im.resize((nwidth,20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
    #     wleft = int(round(((IMAGE_SIZE - nwidth)/2),0)) 
    #     newImage.paste(img, (wleft, 4)) 
    
    

    tv = list(newImage.getdata()) 
    
    
    tva = [ (255-x)*1.0/255.0 for x in tv] 
    return tva
    

def main(argv):
    imvalue = imageprepare(argv)
    predictedLetter = predictLetter(imvalue)
    print (predictedLetter) 
    print (np.argmax(predictedLetter, 1))
    
if __name__ == "__main__":
    # Whenever the script runs independently this main function is called
    # sys.argv provides the image path when used from cmd
    main(sys.argv[1])
