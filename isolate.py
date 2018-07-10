import numpy as np
import cv2 as cv
import os
import math

def getCharList(img_path):
	im=cv.imread(img_path)
	imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
	ret, thresh = cv.threshold(imgray, 0, 255, 0)
	im2, cont, hir = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

	cont.sort(key=lambda x: cv.boundingRect(x)[0], reverse=False)
	cont.pop(0)

	i = 0
	imgList = []
	for c in cont:
		x,y,w,h = cv.boundingRect(c)

		inc_h = 0
		inc_w = 0
		w_is_gre = False
		if w>h:
			# offset = int(w/2)
			size = w
			# inc_w = 1
			w_is_gre = True
		else:
			# offset = int(h/2)
			size = h
			# inc_h = 1

		if size < 200:
			continue

		# roi = im[center[0]-offset:center[0]+offset, center[1]-offset:center[1]+offset]

		roi = imgray[y:y+h, x:x+w]

		blank_image = np.zeros((size,size), np.uint8)
		blank_image[:,:] = 255

		# print(roi.shape)
		# print(blank_image.shape)
		# print(y,y+h)
		# print(x,x+w)

		center = ( int(math.ceil(blank_image.shape[0]/2)), int(math.ceil(blank_image.shape[1]/2)) )
		half_h = int( math.ceil(h/2))
		half_w = int( math.ceil(w/2))
		if 2*half_h > roi.shape[0]:
			inc_h = -1
		elif 2*half_h < roi.shape[0]:
			inc_h = 1

		if 2*half_w > roi.shape[1]:
			inc_w = -1
		elif 2*half_w < roi.shape[1]:
			inc_w = 1
		

		# print(center)
		# print(half_h)
		# print(half_w)

		blank_image[center[0]-half_h:center[0]+half_h+inc_h, center[1]-half_w:center[1]+half_w+inc_w] = roi

		imgList.append(blank_image)

		cv.imwrite(str(i)+"_img.png", blank_image)
		i+=1

	# cv.waitKey()
	return imgList

# getCharList("multi-test.png")