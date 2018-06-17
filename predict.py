import sys
import tensorflow as tf
from PIL import Image,ImageFilter
import numpy as np

# Total number of charachters to predict from
TOTAL_ELEMENTS = 36

# Size of reduced image
IMAGE_SIZE = 28

# For printing
folders = [ "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
             "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
             "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

# This is the function to predict the letter
def predictLetter(imvalue):

    # Initiate the tensorflow graph
    tf.reset_default_graph()

    # Empty tensorflow components
    x = tf.placeholder(tf.float32, shape=[None, 784])
    W = tf.Variable(tf.truncated_normal([784, TOTAL_ELEMENTS]), dtype=tf.float32, name="weights_0")
    b = tf.Variable(tf.truncated_normal([TOTAL_ELEMENTS]), dtype=tf.float32, name="bias_0")
    y = tf.nn.softmax(tf.matmul(x, W) + b)
    
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
   
        prediction=tf.argmax(y,1)
        index = prediction.eval(feed_dict={x: [imvalue]}, session=sess)
        return folders[index[0]]

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
    print (folders[predictedLetter[0]] )
    
if __name__ == "__main__":
    # Whenever the script runs independently this main function is called
    # sys.argv provides the image path when used from cmd
    main(sys.argv[1])
