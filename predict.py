import sys
import tensorflow as tf
from PIL import Image,ImageFilter

TOTAL_ELEMENTS = 36

def predictLetter(imvalue):
    tf.reset_default_graph()

    x = tf.placeholder(tf.float32, shape=[None, 784])
    W = tf.Variable(tf.truncated_normal([784, TOTAL_ELEMENTS]), dtype=tf.float32, name="weights_0")
    b = tf.Variable(tf.truncated_normal([TOTAL_ELEMENTS]), dtype=tf.float32, name="bias_0")
    y = tf.nn.softmax(tf.matmul(x, W) + b)
    
    # init_op = tf.initialize_all_variables()
    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver()

    tf_config = tf.ConfigProto(
        device_count = {'GPU': 0})

    with tf.Session(config=tf_config) as sess:
        sess.run(init_op)
        saver.restore(sess, "./Model/model.ckpt")
   
        prediction=tf.argmax(y,1)
        return prediction.eval(feed_dict={x: [imvalue]}, session=sess)

def getImage(image):

    im = image.convert('L')
    width = float(im.size[0])
    height = float(im.size[1])
    newImage = Image.new('L', (28, 28), (255)) 
    
    if width > height: 
        nheight = int(round((20.0/width*height),0)) 
        if (nheight == 0): 
            nheight = 1
        img = im.resize((20,nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wtop = int(round(((28 - nheight)/2),0)) 
        newImage.paste(img, (4, wtop)) 
    else:
        
        nwidth = int(round((20.0/height*width),0)) 
        if (nwidth == 0): 
            nwidth = 1
         
        img = im.resize((nwidth,20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wleft = int(round(((28 - nwidth)/2),0)) 
        newImage.paste(img, (wleft, 4)) 
    
    

    tv = list(newImage.getdata()) 
    
    
    tva = [ (255-x)*1.0/255.0 for x in tv] 
    return tva

def imageprepare(argv):

    im = Image.open(argv).convert('L')
    width = float(im.size[0])
    height = float(im.size[1])
    newImage = Image.new('L', (28, 28), (255)) 
    
    if width > height: 
        nheight = int(round((20.0/width*height),0)) 
        if (nheight == 0): 
            nheight = 1
        img = im.resize((20,nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wtop = int(round(((28 - nheight)/2),0)) 
        newImage.paste(img, (4, wtop)) 
    else:
        
        nwidth = int(round((20.0/height*width),0)) 
        if (nwidth == 0): 
            nwidth = 1
         
        img = im.resize((nwidth,20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wleft = int(round(((28 - nwidth)/2),0)) 
        newImage.paste(img, (wleft, 4)) 
    
    

    tv = list(newImage.getdata()) 
    
    
    tva = [ (255-x)*1.0/255.0 for x in tv] 
    return tva
    


def main(argv):
    imvalue = imageprepare(argv)
    predictedLetter = predictLetter(imvalue)
    print (predictedLetter[0]) 
    
if __name__ == "__main__":
    main(sys.argv[1])
