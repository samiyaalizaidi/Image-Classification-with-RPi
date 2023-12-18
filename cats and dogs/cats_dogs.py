import os
os.environ['HDF5_DISABLE_VERSION_CHECK'] = '2'

# IMPORT THE MODULES
import cv2
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# LOAD THE PRETRAINED MODEL
model = tf.keras.models.load_model('CATS_DOGS_MODEL_1.h5', compile=False)

# Load the video file
video_path = 'output_cd_pi.h264'
cap = cv2.VideoCapture(video_path)

# Get the fps of the path
fps = cap.get(cv2.CAP_PROP_FPS)

# counter
counter = 0

while True:
	ret, frame = cap.read()
	
	if not ret:
		break

	counter += 1

	print(counter, end= ' ')

	# Resize the image
	input_size = (224, 224)
	resized_frame = cv2.resize(frame, input_size)

	img_array = img_to_array(resized_frame)
	img_array = tf.expand_dims(img_array, 0)

	predictions = model.predict(img_array)

	if int(round(predictions[0][0])) == 0:
		print('cat', end=' ')
	else:
		print('dog', end=' ')

	print(predictions[0][0])

cap.release()
cv2.destroyAllWindows()