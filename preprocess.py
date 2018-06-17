import os
from PIL import Image
import numpy as np

def get_and_preprocess(path):
	im = Image.open(path)
	im_g = im.convert('L')
	im_g_a = np.asarray(im_g)
	mask = im_g_a < 255
	coords = np.argwhere(mask)
	y0,x0 = coords.min(axis=0)
	y1,x1 = coords.max(axis=0)

	len_x = (x1-x0)
	len_y = (y1-y0)

	center = ( x0 + len_x/2 , y0 + len_y/2 )

	if( len_x > len_y ):
		im_crop = im.crop( (x0, center[1] - (len_x/2) , x1 , center[1] + (len_x/2) ) )
	else:
		im_crop = im.crop( ( center[0] - (len_y/2), y0, center[0] + (len_y/2), y1 ) )
	
	return im_crop


path = "Data/training_preprocess/"
new_path = "Data/training/"

folders = os.listdir(path)
# print (folders)

for folder in folders:
	if not os.path.exists(new_path+folder):
    		os.makedirs(new_path+folder)

	files = os.listdir(path + folder)

	for file in files:
		img_path = path + folder + "/" + file
		# print (img_path)

		im = get_and_preprocess(img_path)

		im.save(new_path + folder + "/" + file)


