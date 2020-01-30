import csv
import matplotlib.pyplot as plt
import numpy as np
import pdb

csv_path = "..\\5_rakshith_data\\2_both_edge correction2\\driving_log.csv"
rel_path = '..\\5_rakshith_data\\2_both_edge correction2\\IMG\\'
trim_path = True
trim_whitespace = False
full_windows_path = True

rows=[]
with open(csv_path, 'r') as f:
    content = csv.reader(f, delimiter=',')
    for row in content:
        rows.append(row)

rows = rows[1:]  # Remove the title line.
include_lr = True
turn_value = 2
right_cam_bias = 1.3
left_cam_bias = 1.3
def data_generator(rows, batch_size, include_lr, turn_value, trim_path):
    ctr = 0
    while True:
        for ind in range(0, len(rows), batch_size):

            rows_batch = training_rows[ind:ind+batch_size]
            images = []
            steering = []
            for row in rows_batch:
                ctr+=1
                if (ctr%2000 == 0):
                    print("\nExample Count: ", ctr)
                
                # Regular Images                    
                path = rel_path + row[0].split('\\')[-1]
                if include_lr:
                    path_l = rel_path + row[1].split('\\')[-1]
                    path_r = rel_path + row[2].split('\\')[-1]

                # Image PRocessing
                image = plt.imread(path)
                images.append(image)
                if include_lr:
                    image = plt.imread(path_l)
                    images.append(image)
                    image = plt.imread(path_r)
                    images.append(image)

                # Processing Steering Angle
                steering.append(float(row[3]))
                if include_lr:
                    steering.append(float(row[3]) + turn_value + left_cam_bias)
                    steering.append(float(row[3]) - turn_value - right_cam_bias)

                # Mirrored Images
                path = rel_path + row[0].split('\\')[-1]
                if include_lr:
                    path_l = rel_path + row[1].split('\\')[-1]
                    path_r = rel_path + row[2].split('\\')[-1]
                image = plt.imread(path)
                image_mirror = np.fliplr(image)
                images.append(image_mirror)
                if include_lr:
                    image = plt.imread(path_l)
                    image_mirror = np.fliplr(image)
                    images.append(image)
                    image = plt.imread(path_r)
                    image_mirror = np.fliplr(image)
                    images.append(image)    
                steering.append(-float(row[3]))
                if include_lr:
                    # When the image is Mirrored, the L camera becomes the R camera and vice-versa. 
                    # So changing the order.
                    #steering.append(-(float(row[3]) + turn_value) )
                    #steering.append(-(float(row[3]) - turn_value) )
                    steering.append(-(float(row[3]) - turn_value - left_cam_bias) )
                    steering.append(-(float(row[3]) + turn_value + right_cam_bias) )

            #print(np.shape(images))
            #print(np.shape(steering))
            X_train = np.array(images)
            y_train = np.array(steering)
            yield shuffle(X_train, y_train)



from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
training_rows, validation_rows = train_test_split(rows, test_size=0.2)
batch_size = 32
training_gen = data_generator(training_rows, batch_size, include_lr, turn_value, trim_path)
validation_gen = data_generator(validation_rows, batch_size, include_lr, turn_value, trim_path)


from keras.models import Sequential
from keras.layers import Flatten, Dense, Activation, Conv2D, Dropout, Lambda, Cropping2D, MaxPooling2D

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))

model = Sequential()

# CNN
model.add( Lambda(lambda x: x/255 - 0.5, input_shape=[160,320,3]) )  # Dims: 160 x 320 x 3
model.add( Cropping2D( ((70, 25), (0,0)) ) )  # Dims: 65 x 320 x 3

model.add(Conv2D(6, (9,19), padding='valid', strides=1))  # Dims: 57 x 300 x 6
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))  # Dims: 28 x 150 x 6
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Conv2D(16, (4,4), padding='valid', strides=1))  # Dims: 25 x 74 x 16
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))  # Dims: 12 x 37 x 16
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Conv2D(32, (3,3), padding='valid', strides=1))  # Dims: 10 x 72 x 32
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))  # Dims: 5 x 36 x 32
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(5))
model.add(Dense(1))


model.compile(loss='mse', optimizer='adam')
import math
# model.fit(X_train, y_train, validation_split = 0.2, shuffle=True, epochs=3, batch_size=128)
model.fit_generator(\
    training_gen, \
    steps_per_epoch=math.ceil(len(training_rows)/batch_size), \
    validation_data=validation_gen, \
    validation_steps=math.ceil(len(validation_rows)/batch_size), \
    epochs=2, \
    verbose=1)  # This will perform the next() opperation on the generator when ready.

model.save('network.net')
