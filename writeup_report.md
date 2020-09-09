# **Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # "Image References"

[image1]: ./examples/convolution.jpg "Model Visualization"
[image2]: ./examples/model.jpg "model architecture"
[image3]: ./examples/center.jpg "center image"
[image4]: ./examples/center_fliplr.jpg "flipped Image"


### Model Architecture and Training Strategy

#### 1. Create Dataset

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image3]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover. I repeated this process on track two in order to get more data points.

After the collection process, I had 11284 number of data points(only use center image(X_train) and steering(label)). 


Then I randomly shuffled the data set and put 20% of the data into a validation set. 

To augment the data set, I also flipped images in training set,  this would double the training data. For example, here is an image that has then been flipped:

![][image4]

I used this training data for training the model. The validation set helped determine if the model was over or under fitting.  

#### 2. Solution Design Approach

First, I cropped and normalized the data.  Because the top and bottom area is not useful for training. Then I used a convolution neural network model similar to the LeNet. I thought this model might be appropriate convolution neural network is good method to solve computer vision problem.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I used an adam optimizer so that manually training the learning rate wasn't necessary.  I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I added two dropout layers the model. I found that 3 epochs is enough for this model(the loss of validation is close to training validation).

The final step was to run the simulator to see how well the car was driving around track one. The car was out of track at a few spots when using the give dataset. I thought it is because data is not enough and used the data collected by myself, and it really works.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 3. Final Model Architecture

Here is a visualization of convolution layers.

![alt text][image1]

Here is the whole architecture of model.

![alt text][image2]


