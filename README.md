# Behavioral Cloning Project

Overview
---
The broad steps of this project are:

* Using the simulator to collect data of good driving behavior 
* Designing, training and validating a model that predicts a steering angle from image data
* Using the model to drive the vehicle autonomously around the first track in the simulator.


[//]: # (Image References)

[image1]: ./images/dataset.png "Dataset"
[image2]: ./images/camera_angles_left.jpg "Camera Left"
[image3]: ./images/camera_angles_center.jpg "Camera Center"
[image4]: ./images/camera_angles_right.jpg "Camera Right"
[image5]: ./images/crop_orig.jpg "Original Image"
[image6]: ./images/crop_cropped.jpg "Cropped Image"
[image7]: ./images/correction_center.jpg "Correction"
[image8]: ./images/unreflected.jpg "Unreflected"
[image9]: ./images/reflected.jpg "Reflected"


Data Collection
---

#### i. Two Types of Driving Data

The First step would be to collect good training data. For the purpose of this project the data is camera images taken along the length of the track using the camera within the car. There were two ways in which I drove the car while collecting data:

* One was the straightforward and ideal scenario, wherein I ensured the car was centered on the road and maintained a good steering angle throughout the track
* The second was that of corrective drining. Here I would position the car such that it was severely off-centered on the road. From this edge/side position I would press the record button and recenter the car. A camera image taken from the car when it was highly offcentered on purpose (to train it for correction) is given below:


![alt text][image7] 


Steering angles associated with each image were already in the normalized range of [-1, +1].

A sample of the csv file holding the camera image locations and the corresponding steering angle is provided below:

![alt text][image1]

#### ii. Right and Left Camera Angles

For each of the images captured from within the car, there were two other camera angles that were recorded. One that was placed slightly to the right and the other that was positioned slightly to the left. Taking such images helped in two ways - They helped provide driving data for the car in the case it was slightly off centered and also helped provide a more generalized dataset. Thus, based on the former reasoning, it helped the car recenter itself when offset slightly from the center, and by the latter reasoning it ensured lesser chances of overfitting.

The left, center and right camera angles for the same point in the track are shown below:
![alt text][image2] ![alt text][image3] ![alt text][image4] 


#### iii. Augmented Images

The track mainly consisted of left turns and only one major right turn existed. To ensure the model generalizes well, all the images that were collected were reflected and augmented to the original dataset. The steering angle associated with such reflected images was the negated value of the original value. An example of the original and reflected image is given below:

![alt text][image8] ![alt text][image9] 



Network Design and Training
---

#### i. Pre-processing

As most datasets, this dataset also consisted of a couple of pre-processing steps:

* First, the images were normalized to ensure zero means. This was done in keras using a ```Lambda layer```. 

* Then, the images were cropped to ensure most of the surroundings were cropped out and the main object within the image was the road. This was done to ensure the network did not learn wrong details picked up from surrounding objects. For this, the ```Cropping2D layer``` was used.

The original and cropped image is shown below:

![alt text][image5] ![alt text][image6] 

#### ii. Network 1 [ Single Layer NN/Perceptron -- Failed ]

I wanted to start simple and build layers and complexity into the network gradually. The first "netowork" that I tried was a simply a single layer perceptron using the ```Flatten()``` fuction. 

```
# Single Layer  NN
model.add( Lambda(lambda x: x/255 - 0.5, input_shape=[160,320,3]) )
model.add( Cropping2D( ((60, 20), (0,0)) ) )
model.add(Flatten())
model.add(Dense(1))
```

Since the output required was a single number (the steering angle), an output layer of ```Dense(1)``` was used. I continued to use the Adam optimizer here as well (as I did from previous projects). The loss function was set to use the MSE (mean square error). This network was applied on the normalized images. Around ```10 EPOCHS``` were used while training the network. A batch size of 128 images was used.


It was seen that the car showed some signs of attempting to stay on the road. There were constant steering adjustments. After a short distance however the car would go off the road. 


The reason for the sudden off roading seemed to arise from the surrounding noise in the images caused by the trees and bushes. Thus, I added a Lambda layer to crop off ```60 pixels``` from the top and ```20 pixels``` off the bottom. This prevented the car from suddenly turing off the road at the previous point of failure, and helped it make slow and highly oscillatory progress till the first turn. 


#### iii. Network 2 [ Multi-Layer NN -- Succeeded ]

This network was applied on the normalized and cropped (this time ```70 pixels``` from the top and ```25 pixels``` from the bottom) images. To add a little more scope for learning into the network, I added a few layers of fully connected layers. Thus the network took on the following shape:

```
# MultiLayer NN
model.add( Lambda(lambda x: x/255 - 0.5, input_shape=[160,320,3]) )
model.add( Cropping2D( ((70, 25), (0,0)) ) )
model.add(Flatten())
model.add(Dense(25))
model.add(Dense(25))
model.add(Dense(25))
model.add(Dense(1))
```

This network was trained for ```10 EPOCHS``` again, and it performed quite well. It ensured the car almost navigated around the first turn.


To improve it, I included the augmented dataset of left and right camera images. To each of these images I added a steering bias, such that the car would be taught to turn slightly more to the right for images taken from the left camera and vice-versa. Inclusion of this set helped the car navigate all the way till the bridge!

The next change I made to the model was altering the number of EPOCHS and turning bias. I also included the data collected from the manual "correction drive" that I had perfomed. Finally, I used ```2``` EPOCHS, used a small steering bias and also changed the batch size to ```64```. To my surprise, this ensured the car navigated the entire path. It was even able to navigate the entire path within Track2 without driving off or falling off the road despite being trained only on track1!


The car was still mildly oscillatory. To address this, I tried different sizes of hidden layers. However this did not seem to have much more of an effect. Thus I began working on Network 3.


#### iv. Network 3 [ Convolution NN -- Succeeded, smoother]

The goal of implementing this network was to get the car to behave more smoothly around the track. Although I could have referred to other network architectures designed say by Nvidia for the purpose of a self driving car, I wanted to develop a CNN from scratch. It involved playing around with a different number of parameters:

* The number of convolution layers
* The dimensions of the path/kernel
* Dropout rate
* Number of Fully Connected Layers
* Size of Fully Connected Layers


The entire augemnted dataset (```Original Center images``` + ```Reflected Center Inages``` + ```Left/Right Images with Steering Bias``` + ```Reflected Left/Right Images``` + ```Corrective dataset of images [which itself had left/center/right images and their reflections]```) was used to train this network. Finally, after a lot of testing and tweaking, I was able to train a satisfactory network with the following architecture:

```
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
```
  
It required 2 EPOCHS to train and a batch size of ```32``` was used. The turning bias was increased slightly for this model.


Results: Autonomous Driving
---

The previous section discussed how the model was trained, tested and finally arrived at. Here I have provided the videos showing the analysis and description provided in the previous section. 


#### i. Network 2 [ Multi-Layer NN -- Succeeded ]

* **Track1 (Trained on Track1): ** Video available at: ```./videos/MultiNN_track1.mp4``` {Exluded from submission due to upload size constraint}

* **Track2 (Trained only on Track1): ** Video available at: ```./videos/MultiNN_track2.mp4``` {Exluded from submission due to upload size constraint}


**Observations and Intuitions:**
* This model succeeded in navigating both track1 and track2 despite being trained only on track1. It was however slighlty shaky on track1 and highly oscillatory on track2. 
* The network was able to learn about the left and right road margins and navigated sufficiently well to stay within their bounds. 
* Thus, on track2 it was able to stay on the road by merely identifying the road boundaries/edges. To ensure that the car stayed on one side of the road on track2, the network will need to be trained on track2 as well such that it is shown how to stay on one side of the road. This would require quite a lot of tweaking, retraining and possibly may not be do-able with just a multi-layer neural network. A more complex archetecture like a CNN might need to be used.

#### ii. Network 2 [ Multi-Layer NN -- Succeeded ]

* **Track1 (Trained on Track1): ** Video available at: ```./videos/.CNN_track1.mp4``` {Included for submission}


**Observations and Intuitions:**
* This model succeeded in navigating track1 in a very smooth manner.
* To make it work well with track2, it will need to be fed data from track2 as well.

