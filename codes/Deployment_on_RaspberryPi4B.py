import os
import RPi.GPIO as GPIO
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
import cv2
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

green_led = 18
red_led = 20
GPIO.setmode(GPIO.BOARD)
GPIO.setup(led_green,GPIO.OUT)
GPIO.setup(led_red,GPIO.OUT)


model = load_model('/home/pi/Desktop/Covid19_detection/covid_classifier_final')
test_file = '/home/pi/Desktop/Covid19_detection/test3.jpg'
test_img = load_img(test_file, target_size = (150,150))
test_img = np.asarray(test_img)
test_img = np.expand_dims(test_img,axis=0)
test_img = test_img / 255
prediction = model.predict(test_img)[0]

if round(prediction[0]) == 1:
      print("Covid Negative ")
      GPIO.output(green_led,GPIO.HIGH)
    
else:
      print("Covid Positive")
      GPIO.output(red_led,GPIO.HIGH)
 
 
 
 
 
