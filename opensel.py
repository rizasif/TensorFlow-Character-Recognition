from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from kivy.clock import Clock
from kivy.garden.filebrowser import FileBrowser
from os.path import sep, expanduser, isdir, dirname
import sys

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
		# raise ValueError("No File Selected")
		sys.exit("No File Selected")

	# called when browser selects file
	def _fbrowser_success(self, instance):
		# global global_mainApp
		print (instance.selection)
		selected_file_path = instance.selection
		sys.exit(str(selected_file_path[0]))
		
		# print("Main App Value is: ", type(global_mainApp))
		
		# Browser throws the selected file string as error
		# So that the main app can catch it
		# raise ValueError(str(selected_file_path[0]))

# The main function
if __name__ == '__main__':
	# TestApp().run()
	BrowseApp().run()