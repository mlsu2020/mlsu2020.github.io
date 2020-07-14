# Road Segmentation and Turn Classification Using Supervised and Unsupervised Learning

## Contents

## Introduction
Autonomous vehicles need to know if they should be turn left, right, or go straight. The goal of our project was to create a machine learning model that can make these classifications based on road images.

## Dataset
Our dataset consisted of images and state information from Georgia Tech's [AutoRally Project](https://autorally.github.io/), a platform for self-driving vehicle research. Robotic cars from this project record a first-person video the area in front of them while driving around a dirt track. We selected a subset of this dataset which consists of two laps (approximately 90 seconds) of driving data. The driving data consists of images taken from the vehicle's perspective at 40Hz as well as state estimations (such velocities in the vehicles frame and global positions) based on GPS and IMU measurements.

Below we display the state estimates for the longitudinal velocity, lateral velocity, yaw rate, front and rear wheel speeds, yaw orientation, and global X and Y positions over the 2 laps of driving data selected for the prject. We also display a couple of sample images. 

<div style="display: flex; flex-direction: row; justify-content: space-evenly; width: 100%;"><img src="images/states.png" width="100%"></div>


<div style="display: flex; flex-direction: row; justify-content: space-evenly; width: 100%;">
  <img src="images/img47.png" width="40%">
  <img src="images/img676.png" width="40%">
</div>

## Methods
Our project took the following approach to classifying whether the autorally car should turn left, right, or go straight:

1.  Generate a reduced representation of the road scene by segmenting the road from the image. A car's traversable space is limited by the shape of the road they're on, so we figured the **road map** would provide clear training data for a neural network.
    <div style="display: flex; flex-direction: row; justify-content: space-evenly; width: 100%;"><img src="images/img47.png" width="40%"><img src="images/img47_o.png" width="40%"></div>
2.  Use dynamics data to label each image as going left, right, or straight. This is done by filtering and thresholding the vehicle's yaw velocity at the time each image was captured to determine the direction the car was turning.
3.  Train a neural network on the labelled images to classify the segmented road maps to the correct road segment types (right turn, straight, or left turn).</div>


## Road Segmentation
Road detection can be a difficult problem since the appearance of roads vary depending on lighting conditions, road texture, and the presence of obstacles. The AutoRally data provided us a simplified scene to work with: the images were taken during daytime on a dirt road with no obstacles.

We looked to two clustering algorithms to help us segment the road from the image: DBSCAN and K-Means clustering

### DBSCAN Approach
We started by trying the [DBSCAN clustering algorithm](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html) on the road images. One of the advantages of DBSCAN is that it can determine the number of clusters on its own. This could be helpful in generalized road detection algorithms since we can't always expect there to be consistent lighting conditions.

This is the overall algorithm we used:
1.  Pre-processing
    * Crop the image to its bottom 2/3rds to ignore the sky and focus on the road.
    * Apply histogram equalization to make the road stand out from the road edges and car hood.
    * Blur the image to even-out the rocky appearance of the road
    <div style="display: flex; flex-direction: row; justify-content: space-evenly; width: 100%;"><img src="images/dbscan_preprocessing.png" width="80%"></div>
2.  Run DBSCAN to get cluster labels. We used the 3rd largest distance from a [K-NN](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html) search for DBSCAN's epsilon parameter.<sup><a href="#ref1">[1]</a></sup>
    <div style="display: flex; flex-direction: row; justify-content: space-evenly; width: 100%;"><img src="images/dbscan_output.png" width="60%"></div>

3. Use connected component analysis and a 2D Gaussian to select the cluster that has many points in the center of the image, where we assume the road to be.
<div style="display: flex; flex-direction: row; justify-content: space-evenly; width: 100%;">
<img src="images/dbscan_cc.png" width="95%">
</div>

### Results
Despite our pre-processing steps and DBSCAN parameter tuning, we found the clusterings produced by DBSCAN to still be too sensitive to road aberrations like grass and rocks, and sometimes the road wouldn't be selected.
<div style="display: flex; flex-direction: row; justify-content: space-evenly; width: 100%;">
  <img src="images/dbscan_wrong.png" width="60%">
</div>

Roughly half the road maps would have errors, so we looked to using K-Means

## Road Segmentation Using K-Means
To apply K-Means clustering to the road segmentation problem, we made the guiding assumption that (1) light areas are sky, (2) dark areas are the car hood or road edges, and (3) midtones contain the road. This assumption is reliant on consistent lighting conditions, so it's not generalizable to most road detection problems. For the specific data we have though, it allowed us to perform the clustering with K=3.

1.  Apply Gaussian Blur to smooth the road
2.  Run K-Means clustering on the RGB image with K=3.
    <div style="display: flex; flex-direction: row; justify-content: space-evenly; width: 100%;"><img src="images/kmeans_k3.png" width="85%"></div>
3.  Assuming the midtone contains the road, extract it with a mask, and perform binary opening and closing to disconnect thin regions.
    <div style="display: flex; flex-direction: row; justify-content: space-evenly; width: 100%;"><img src="images/kmeans_mask.png" width="60%"></div>
4.  Apply connected component analysis and select the largest cluster as the road. Apply further connected component analysis and flood-filling to fill in gaps in the road.
    <div style="display: flex; flex-direction: row; justify-content: space-evenly; width: 100%;"><img src="images/kmeans_final.png" width="85%"></div>

### Results
Compared to DBSCAN, only about ~2% of the road maps generated by K-Means had errors in the chosen cluster. Therefore, we used this data in our neural network.

## Neural Network

We chose to use a neural network to perform image classification on the segmented road map images. Since we have already generated our feature information in the form of the segmented roadmap, our neural network only needs to map the spatial information to the three possible labels (left turn, straight, right turn). Therefore we elected to use a series of fully connected layers to map the segmented pixel locations to the three categories. We also used one max pooling layer to reduce the image dimensionality and the number of learnable parameters in our network to speed up training.

### Network Architecture

<div style="display: flex; flex-direction: row; justify-content: space-evenly; width: 100%;"><img src="images/image_class_net.png" width="17%"></div>

The output class scores are then fed into a softmax function for training which assigns a probability of the image belonging to each of the three classes based on the relative weight of the scores. This allows the gradient to be backpropagated through all three output channels. For validation, we simply take the argmax of the three output channels to determine the most likely class the image belongs to.

### Results
#### Training and Validation Accuracy

We trained the network for 20 epochs using the Adam optimizer using a cross entropy loss function. The cross entropy loss incorporates softmax which assigns a probability of the image belonging to each class based on the relative scores of the three output channels. The optimier then adjusts the layer weights to maximize the relative scores of the correct classes. We reserved one third of our dataset for validation to prevent overfitting, and plotted the training and validation accuracies for each epoch. We reach about 96% accuracy on the training data and 87% on the validation set.

The training and validation datasets were generated automatically using the vehicle states information accompanying the images. We filtered the vehicle's yaw velocity and then thresholded it to fall into one of three classes: right turns, straights, or left turns. The thresholded yaw velocity at the time each image was captured was used to provide the image labels for supervised learning.

<div style="display: flex; flex-direction: row; justify-content: space-evenly; width: 100%;"><img src="images/training_acc1.png" width="80%"></div>

#### Validation Examples

We demonstrate the classification results on a sample of validation images. Although the network takes the segmented images as the inputs, we display the raw, unsegmented images for easy visual inspection.

<div style="display: flex; flex-direction: row; justify-content: space-evenly; width: 100%;"><img src="images/image_examples2.png" width="100%"></div>

## Conclusion

## References.
1. <a id="ref1" href="https://iopscience.iop.org/article/10.1088/1755-1315/31/1/012012/pdf" style="display: inline">Determination of Optimal Epsilon Value on
DBSCAN Algorithm to Clustering Data on
Peatland Hotspots in Sumatra</a>
