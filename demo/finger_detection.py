# python3 finger_detections 0

from skimage.morphology import reconstruction
from skimage.morphology import opening

import numpy as np
import cv2
import sys

def handDescomposition(mask):

	#invert color for viz purpose
	mask = 255-mask
		
	hand = opening(mask, np.ones((51,51)))
		
	#top-hat
	fingers = mask-hand
	kernel  = np.ones((3,3), np.uint8) 
	fingers_enh = cv2.erode(fingers, kernel, iterations=5)
		
	#fingers-components above given area-threashold            
	components = cv2.connectedComponentsWithStats(np.uint8(fingers_enh), connectivity=4)
	
	try:
		thd = min(max(components[2][1:, 4])//2,700)
	except:
		thd = 700
	
	return  sum(components[2][1:,4]>thd)


def postProcessMask(mask):
	
	#hole filling
	padded = np.pad(mask, pad_width = 1, mode = "constant", constant_values = 255) 

	seed = np.zeros(padded.shape)
	seed[:,0] = 255
	seed[:,-1] = 255
	seed[0,:] = 255
	seed[-1,:] = 255
		  
	filled = reconstruction(seed, padded)
	filled = filled[1:-1,1:-1]
		
	kernel = np.ones((5,5),np.uint8)
		
	filled = cv2.erode(filled, kernel, iterations=3)
		
	seed = np.ones(filled.shape)*255
	n,m = seed.shape
	seed[n//2,m//2] = 0
		
	# reconstruction
	try:
		filled = reconstruction(seed, filled, method='erosion')
	except:
		pass
		
	return filled


def pixelClassification(image, channels, stats):
	n,m,_ = image.shape
	image_copy = np.ones(image.shape, int)*255
	mask = None
	
	for k in range(channels):
		pos_values = (stats[k][0] - stats[k][1] <= image[:,:,k]) == (image[:,:,k] <= stats[k][0] + stats[k][1])
		image_copy[~pos_values,k] = 0
		#bitwise operator: AND  applied to channels
		if k > 1:     
			mask = mask & image_copy[:,:,k] 
		else:
			mask = image_copy[:,:,k]
	return 255-mask


def predictSkinMask(image, stats):
	image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
	return pixelClassification(image[:,:,1:], 2, stats)
	

def skinStats(image):
	f = lambda x: (np.mean(x), np.std(x))
	
	skin = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

	stats = {}
	
	stats[0] = f(skin[:,:,1])
	stats[1] = f(skin[:,:,2])

	return stats


def initSystem(src):
	return cv2.VideoCapture(src)
	

def main():

	try:
		src = int(sys.argv[1])
	except:
		print('Default camera source')
		src = 0

	video = initSystem(0)

	#Skin Characterization
	while True:
		ret, frame = video.read()
		frame=cv2.flip(frame, 1)

		text_skin = 'Put ONLY HAND SKIN in the box for skin characterization and press: t'		
		cv2.putText(frame,text_skin,(100,70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv2.LINE_AA)
		cv2.rectangle(frame,(100,100),(300,300),(0,255,0),0)

		cv2.imshow('frame',frame)

		if cv2.waitKey(1) == ord("t"):
			box=frame[100:300, 100:300]
			stats = skinStats(box)
			break

	#Finger Detection
	while True:
		ret, frame = video.read()
		
		frame = cv2.GaussianBlur(frame, (5,5), 0)
		frame = cv2.flip(frame, 1)

		text_skin = 'Put Hand in the box for Finger Detection.'		
		cv2.putText(frame,text_skin,(100,70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv2.LINE_AA)
		
		text_skip = 'Press q to quit.'		
		cv2.putText(frame,text_skip,(100,570), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv2.LINE_AA)
		cv2.rectangle(frame,(100,100),(500,500),(0,255,0),0)

		box=frame[100:500, 100:500]

		mask = predictSkinMask(box, stats)
		mask = postProcessMask(mask)
		fingers = handDescomposition(mask)

		cv2.putText(frame,str(fingers),(450,450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 2, cv2.LINE_AA)

		cv2.imshow('mask', mask)
		cv2.imshow('frame',frame)

		if cv2.waitKey(1) == ord("q"):
			break


	cv2.destroyAllWindows()
	video.release()


if __name__ == '__main__':
	main()
