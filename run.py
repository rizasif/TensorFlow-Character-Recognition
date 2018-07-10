from kivy.config import Config
Config.set('graphics', 'width', '600')
Config.set('graphics', 'height', '300')
Config.set('graphics', 'resizable', False)

from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from kivy.clock import Clock
from kivy.garden.filebrowser import FileBrowser
from os.path import sep, expanduser, isdir, dirname
import multiprocessing as mp
# from pathos.multiprocessing import ProcessingPool
# import threading
import time
import sys
import traceback
from PIL import Image
import os
import pickle
import subprocess

# os.environ['KIVY_GL_BACKEND'] = 'angle_sdl2'

# sys.setrecursionlimit(100000)

import predictmultiple as predict
# import predict
# import createModel

# Global variables to output messages
selected_file_path = "No File Selected"
error_message = "[color=f44336]{}[/color]"
status_message = "{}"
good_message = "[color=4CAF50]{}[/color]"

global is_training
is_training = False

# Global reference to main app
global_mainApp = None

# this class represents the browser to select file 
class BrowseApp(App):
	MainApp = None

	# sets main app of browser
	def setMainAPP(self, MainAPP):
		self.MainApp = MainAPP

	# called when browser opens
	def build(self):
		user_path = dirname(expanduser('~')) + sep + 'Documents'
		browser = FileBrowser(select_string='Select',
							  favorites=[(user_path, 'Documents')])
		browser.bind(
					on_success=self._fbrowser_success,
					on_canceled=self._fbrowser_canceled)
		return browser

	# called when browser cancelled
	def _fbrowser_canceled(self, instance):
		print ('cancelled, Close self.')
		raise ValueError("No File Selected")

	# called when browser selects file
	def _fbrowser_success(self, instance):
		global global_mainApp
		print (instance.selection)
		selected_file_path = instance.selection
		
		print("Main App Value is: ", type(global_mainApp))
		
		# Browser throws the selected file string as error
		# So that the main app can catch it
		raise ValueError(str(selected_file_path[0]))

# A class to handel parallel processing
# Used for opneing browser
class Process(mp.Process):
	def __init__(self, *args, **kwargs):
		mp.Process.__init__(self, *args, **kwargs)
		self._pconn, self._cconn = mp.Pipe()
		self._exception = None

	def run(self):
		try:
			mp.Process.run(self)
			while self.is_alive():
				return pickle.dumps(self)
			# mp.Process.start(self)
			self._cconn.send(None)
		except Exception as e:
			tb = traceback.format_exc()
			self._cconn.send((e, tb))
			# raise e  # You can still rise this exception if you need to

	@property
	def exception(self):
		if self._pconn.poll():
			self._exception = self._pconn.recv()
		return self._exception

# This function is called when "Select File" Button is pressed
def btn_browse(pos):
		global global_mainApp
		global selected_file_path

		print ('pos: printed from root widget: {pos}'.format(pos=pos))
		# subprocess.call(['python', 'opensel.py'])
		# result = subprocess.check_output(['python', 'opensel.py'])
		p = subprocess.getoutput(['python', 'opensel.py'])
		# result = p.poll()
		sp = p.split()
		print("Output: ", sp[len(sp)-1])
		path = sp[len(sp)-1]


		# app = BrowseApp()
		# # app.setMainAPP(global_mainApp)

		# # p = mp.Pool()
		# # p.apply_async(app.run)
		# # p.close()
		# # p.join()
		
		# p = Process(target=app.run)
		# p.start()
		# # p.run()
		# p.join()

		# # pool = mp.Pool()
		# # pool.apply_async(p)
		# # pool.close()
		# # pool.join()

		# while p.is_alive():
		# 	print("Process is alive")
		# 	time.sleep(0.5)

		# # p = ProcessingPool().map(app.run, [])
		# # time.sleep(100)

		# # while p.is_alive():
		# # 	# print("Process is alive")
		# # 	time.sleep(0.5)

		# path = ""
		# if p[0].exception:
		# 	print("found exception")
		# 	error, traceback = p.exception
		# 	# print "Caught Exception: ", str(traceback)
		# 	print ("Caught Error: {}".format(str(error)))
		# 	path = str(error)

		# path = result
		global_mainApp.update(main_label=path, status=good_message.format("Ready!"))
		selected_file_path = path

# This function is called when "Predict" Button is pressed
def btn_predict(pos):
	global global_mainApp
	global selected_file_path
	print ("Strating Prediction")
	try:
		# im=Image.open(selected_file_path)
		# image = predict.getImage(im)
		global_mainApp.update(status=status_message.format("Predicting..."))
		predictedLetter = predict.predictMultiple(selected_file_path)
		result = "PREDICTION RESULT = {}".format(predictedLetter)
		global_mainApp.update(status=good_message.format(result))
		
	except IOError:
		# filename not an image file
		global_mainApp.update(status=error_message.format("Invalid Image or Path"))

# This function is called when "Train" Button is pressed
def btn_retrain(pos):
	global is_training
	if not is_training:
		global global_mainApp
		global_mainApp.update(status=status_message.format("Training..."))
		p = subprocess.Popen(['python', 'createModel.py'])
		is_training = True
	else:
		global_mainApp.update(status=status_message.format("Model is still training"))
	# result = p.poll()
	# sp = p.split()
	# print("Output: ", sp[len(sp)-1])
	# path = sp[len(sp)-1]
	# global_mainApp.update(status=good_message.format(path))


# This is the class that represents the main screen and its buttons
class RootWidget(BoxLayout):
	MainLabel = Label(text=selected_file_path)
	ErrorLabel = Label(text=error_message)

	def __init__(self, MainApp, **kwargs):
		global global_mainApp
		
		self.MainApp = MainApp
		global_mainApp = self.MainApp

		center = (self.get_center_x(), self.get_center_y())
		print (str(center))

		super(RootWidget, self).__init__(**kwargs)
		self.orientation = 'vertical'
		self.MainLabel=Label(text=selected_file_path)
		self.add_widget(self.MainLabel)

		button_layout = BoxLayout(orientation='horizontal', pos=center)
		btn1 = Button(text='Select File', size_hint = (None, 0.5), width=200, pos=center)
		btn1.bind(on_press=btn_browse)
		button_layout.add_widget(btn1)

		btn2 = Button(text='Predict', size_hint = (None, 0.5), width=200, pos = center)
		btn2.bind(on_press=btn_predict)
		button_layout.add_widget(btn2)

		btn3 = Button(text='Train', size_hint = (None, 0.5), width=200, pos = center)
		btn3.bind(on_press=btn_retrain)
		button_layout.add_widget(btn3)

		self.add_widget(button_layout)

		self.ErrorLabel = Label(text=good_message.format("Ready!"), markup=True)
		self.add_widget(self.ErrorLabel)

	def getMainLabel(self):
		return self.MainLabel

	def getOtherLabel(self):
		return self.ErrorLabel

# The globally main app class
class TestApp(App):

	def build(self):
		self.MainWidget = RootWidget(MainApp=self)
		return self.MainWidget

	def on_start(self):
		print ("Starting App")

	def update(self, main_label="", status=""):
		print ("Updating Label: {}".format(main_label))
		if(len(main_label)>0):
			self.MainWidget.getMainLabel().text = main_label
		
		if(len(status)>0):
			self.MainWidget.getOtherLabel().text = status

	def on_pause(self):
		print ("App Paused")
		# Here you can save data if needed
		return True

	def on_resume(self):
		print ("App Resumed")
		# Here you can check if any data needs replacing (usually nothing)
		pass


# The main function
if __name__ == '__main__':
	TestApp().run()
	# BrowseApp().run()