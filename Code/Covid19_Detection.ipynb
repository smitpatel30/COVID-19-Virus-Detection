import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
print(tf.__version__)

from tensorflow.keras.preprocessing.image import ImageDataGenerator
image_gen = ImageDataGenerator(rotation_range=30,
                               width_shift_range=0.1,
                               height_shift_range=0.1,
                               rescale=1/255,
                               shear_range=0.2,
                               zoom_range=0.2,
                               horizontal_flip=True,
                               fill_mode='nearest')
  
image_shape=(150,150,3)
      
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D

model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(3,3),input_shape=(150,150,3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=64, kernel_size=(3,3),input_shape=(150,150,3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=64, kernel_size=(3,3),input_shape=(150,150,3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(128))
model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss ='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()


batch_size = 16

train_image_gen = image_gen.flow_from_directory('/content/drive/MyDrive/Colab Notebooks/Covid 19 mini project/train',
                                                target_size=(150,150),
                                                batch_size=batch_size,
                                                class_mode='binary')

test_image_gen = image_gen.flow_from_directory('/content/drive/MyDrive/Colab Notebooks/Covid 19 mini project/test',
                                               target_size=(150,150),
                                               batch_size=batch_size,
                                               class_mode='binary')

train_image_gen.class_indices 
{'Normal': 1, 'Covid': 0}

import warnings
warnings.filterwarnings('ignore')

results = model.fit_generator(train_image_gen, epochs = 20,
                              steps_per_epoch= 4,
                              validation_data= test_image_gen,
                              )
model.save('covid_classifier_final.h5')
print(results.history.keys())

#  "Accuracy"
plt.plot(results.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epochs')

plt.plot(results.history['accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epochs')
plt.legend(['val_accuracy','accuracy'], loc='upper left')
plt.show()


# "Loss"
plt.plot(results.history['loss'])
plt.plot(results.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['loss','val_loss'], loc='upper left')
plt.show()

#Zipping the final model
!zip -r ./covid_classifier_final.zip ./covid_classifier_final
