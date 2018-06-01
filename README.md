# TensorFlow<sup>TM</sup> EngHand predict (recognise handwriting)


## Installation & Setup

### Overview
This project uses the notMNIST tutorial from the TensorFlow website. The tutorial uses a deep learning model. However the [Chars 74k dataset](http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/) is used for training. Moreover, a GUI is also available for predicting a single handwritten image.

This tensorflow code consists of two scripts: 

1. _createModel.py_ – creates a model model/model.ckpt file based on the  tutorial.
2. *predict.py* – uses the model.ckpt (beginners tutorial) file to predict the correct integer form a handwritten letter in a .png file.
3. *run.py* – uses the Kivy apis to start the GUI.


### Dependencies
The following Python libraries are required.

- Tensorflow - [TensorFlow](https://www.tensorflow.org/)
- Kivy - [Kivy](https://kivy.org/#home).

### Installing TensorFlow
Of course TensorFlow needs to be installed. The [TensorFlow website](https://www.tensorflow.org/versions/master/get_started/index.html) has a good manual. For windows I would suggest the anaconda version. Moreover, the model is built and tested with Python 3.4.

### Installing Kivy
For using the GUI you need Kivy. The [Kivy installation guide](https://kivy.org/#download) provides a comprehensive tutorial.


### Using the GUI
The easiest way is to start the GUI by running:
```
$ python run.py
```
The GUI is self explanatory. First provide an image path and click the predict button. An already trained model will provide results.

### The python scripts
In order to use the cmd version of the predictor, the training images are included in the package. The rest is provided below.

## Running
Running is based on the steps:

1. create the model file
2. create an image file containing a handwritten letter
3. predict the character 

### 1. create the model file
The easiest way is to cd to the directory where the python files are located. Then run:

```python createModel.py```


to create the model based on the notMNIST beginners tutorial.

### 2. create an image file
You have to create a PNG file that contains a handwritten character. The background has to be white and the letter has to be black. Any paint program should be able to do this. Also the image has to be auto cropped so that there is no border around the letter.

### 3. predict the character
The easiest way again is to put the image file from the previous step (step 2) in the same directory as the python scripts and cd to the directory where the python files are located. 

The predict scripts require one argument: the file location of the image file containing the handwritten letter. For example when the image file is test.png’ and is in the same location as the script, run:

```python predict.py test.png```

The script, predict.py, uses the model.ckpt file created by the createModel.py script.


