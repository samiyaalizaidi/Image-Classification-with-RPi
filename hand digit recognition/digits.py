import os
os.environ['HDF5_DISABLE_VERSION_CHECK'] = '2'

# IMPORT THE MODULES
import cv2
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# LOAD THE PRETRAINED MODEL
model = tf.keras.models.load_model('MNIST_CNN.h5', compile=False)

# Load the video file
video_path = 'output_digits_pi.h264'
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

	frame = tf.image.resize(frame, [28, 28])
	frame = tf.image.rgb_to_grayscale(frame, none=None)

	img_array = img_to_array(frame)
	img_array = tf.expand_dims(img_array, 0)

	predictions = model.predict(img_array)

	rounded_Array = np.round(predictions).astypes(int)

	# find the index of the element with value 1
	index = np.where(rounded_Array == 1)[1][0]

	print(counter, ')', 'Prediction =', index, 'Confidence =', predictions[0][index])

cap.release()
cv2.destroyAllWindows()