from scipy.misc import imread, imresize
import numpy as np
from keras.models import load_model
import sys
import isolate
import math

# Size of reduced image
IMAGE_SIZE = 28

# For printing
folders = [ "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

# Total number of charachters to predict from
TOTAL_ELEMENTS = len(folders)

model = load_model('Model/cnn.h5')

# This is the function to predict the letter
def predictLetter(x):
    # print(x)

    # x = imread(x,mode='L')

    #compute a bit-wise inversion so black becomes white and vice versa
    x = np.invert(x)
    #make it the right size
    x = imresize(x,(28,28))
    #convert to a 4D tensor to feed into our model
    x = x.reshape(1,28,28,1)
    x = x.astype('float32')
    x /= 255

    out = model.predict(x)
    # print(folders[np.argmax(out)])
    return folders[np.argmax(out)]

def predictMultiple(x):
    imgList = isolate.getCharList(x)
    num = 0
    for i in range(len(imgList)):
        img = imgList[i]
        letter = predictLetter(img)
        num += int(letter)*math.pow(10,len(imgList)-i-1)
    return int(num)

def main(argv):
    predictedLetter = predictMultiple(argv)
    print (predictedLetter)
    
if __name__ == "__main__":
    # Whenever the script runs independently this main function is called
    # sys.argv provides the image path when used from cmd
    main(sys.argv[1])
