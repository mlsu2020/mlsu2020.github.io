# Road Segmentation and Turn Classification Using Supervised and Unsupervised Learning

## Contents
1.  <a href="#introduction">Introduction</a>
2.  <a href="#dataset">Dataset</a>
3.  <a href="#methods">Methods</a>
4.  <a href="#road-segmentation">Road Segmentation</a>
    1. <a href="#road-segmentation-clustering">Road Segmentation Using Unsupervised Clustering</a>
        * <a href="#road-segmentation-dbscan">DBSCAN</a>
        * <a href="#road-segmentation-kmeans">K-Means</a>
    2. <a href="#road-segmentation-reg">Road Segmentation Using Regularaized Learning</a>
        * <a href="#road-segmentation-ridge-1">Ridge Regression Using Camera Image, Alone</a>
        * <a href="#road-segmentation-ridge-2">Ridge Regression Using Camera Image + Car Dynamics Data</a>
5.  <a href="#supervised-turn-classification">Supervised Turn Classification</a>
    1. <a href="#supervised-labels">Ground Truth Steering Labels Generation</a>
    2. <a href="#supervised-dt">Decision Tree Classification</a>
    3. <a href="#supervised-neural-net">Neural Network Architecture</a>
6.  <a href="#conclusion">Conclusion and Future Work</a>
6.  <a href="#contributions">Contributions</a>
7.  <a href="#references">References</a>

## 1. Introduction
<a id="#introduction"></a> Autonomous vehicles need to know how to identify the shape of the road they're on. Then, from that information, they need to know whether they should turn left, right, or go straight. The goal of our project was to create machine learning models that can detect the road and make steering decisions using road images and the car's dynamics as inputs.

## Dataset
<a id="#dataset"></a> Our dataset consisted of images and state information from Georgia Tech's [AutoRally Project](https://autorally.github.io/)<sup><a href="#ref1">[1]</a></sup>, a platform for self-driving vehicle research. Robotic cars from this project record a first-person video the area in front of them while driving around a dirt track. We selected a subset of this dataset which consists of two laps (approximately 90 seconds) of driving data. The driving data consists of images taken from the vehicle's perspective at 40Hz as well as state estimations (such velocities in the vehicles frame and global positions) based on GPS and IMU measurements.

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
3.  Train a neural network on the labelled images to classify the segmented road maps to the correct road segment types (right turn, straight, or left turn).

## Road Segmentation
<a id="road-segmentation"></a> Road detection can be a difficult problem since the appearance of roads vary depending on lighting conditions, road texture, and the presence of obstacles. The AutoRally data provided us a simplified scene to work with: the images were taken during daytime on a dirt road with no obstacles.

We looked to two clustering algorithms to help us segment the road from the image: DBSCAN and K-Means clustering

### Road Segmentation Using Unsupervised Clustering

<a id="road-segmentation-clustering"></a>

#### DBSCAN
<a id="road-segmentation-dbscan"></a> We started by trying the [DBSCAN clustering algorithm](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html) on the road images. One of the advantages of DBSCAN is that it can determine the number of clusters on its own. This could be helpful in generalized road detection algorithms since we can't always expect there to be consistent lighting conditions.

This is the overall algorithm we used:
1.  Pre-processing
    * Crop the image to its bottom 2/3rds to ignore the sky and focus on the road.
    * Apply histogram equalization to make the road stand out from the road edges and car hood.
    * Blur the image to even-out the rocky appearance of the road
    <div style="display: flex; flex-direction: row; justify-content: space-evenly; width: 100%;"><img src="images/dbscan_preprocessing.png" width="80%"></div>
2.  Run DBSCAN to get cluster labels. We used the 3rd largest distance from a [K-NN](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html) search for DBSCAN's epsilon parameter<sup><a href="#ref2">[2]</a></sup>.
    <div style="display: flex; flex-direction: row; justify-content: space-evenly; width: 100%;"><img src="images/dbscan_output.png" width="60%"></div>

3. Use connected component analysis and a 2D Gaussian to select the cluster that has many points in the center of the image, where we assume the road to be.
<div style="display: flex; flex-direction: row; justify-content: space-evenly; width: 100%;">
<img src="images/dbscan_cc.png" width="95%">
</div>

##### Results
Despite our pre-processing steps and DBSCAN parameter tuning, we found the clusterings produced by DBSCAN to still be too sensitive to road aberrations like grass and rocks, and sometimes the road wouldn't be selected.
<div style="display: flex; flex-direction: row; justify-content: space-evenly; width: 100%;">
  <img src="images/dbscan_wrong.png" width="60%">
</div>

Roughly half the road maps would have errors, so we looked to using K-Means

#### Road Segmentation Using K-Means
<a id="road-segmentation-kmeans"></a> To apply K-Means clustering to the road segmentation problem, we made the guiding assumption that (1) light areas are sky, (2) dark areas are the car hood or road edges, and (3) midtones contain the road. This assumption is reliant on consistent lighting conditions, so it's not generalizable to most road detection problems. For the specific data we have though, it allowed us to perform the clustering with K=3.

1.  Apply Gaussian Blur to smooth the road
2.  Run K-Means clustering on the RGB image with K=3.
    <div style="display: flex; flex-direction: row; justify-content: space-evenly; width: 100%;"><img src="images/kmeans_k3.png" width="85%"></div>
3.  Assuming the midtone contains the road, extract it with a mask, and perform binary opening and closing to disconnect thin regions.
    <div style="display: flex; flex-direction: row; justify-content: space-evenly; width: 100%;"><img src="images/kmeans_mask.png" width="60%"></div>
4.  Apply connected component analysis and select the largest cluster as the road. Apply further connected component analysis and flood-filling to fill in gaps in the road.
    <div style="display: flex; flex-direction: row; justify-content: space-evenly; width: 100%;"><img src="images/kmeans_final.png" width="85%"></div>

##### Results
Compared to DBSCAN, only about ~2% of the road maps generated by K-Means had errors in the chosen cluster. Therefore, we used this data in our neural network.

<div style="display: flex; flex-direction: row; justify-content: space-evenly; width: 100%;">
  <img src="images/kmeans_output.png" width="60%">
</div>

### Road Segmentation Using Regularaized Learning
<a id="road-segmentation-reg"></a> Predicting the road segmentation is important in real-time applications, as it projects a future planning space for the autonomous vehicle to plan and control. Unsupervised learning is computationally prohibitive to use in a real-time application of automonous driving, where the envrionment is dynamic and the onboarding computation resource is limited. On the other hand, supervised learning can provide a real-time road detection solution, once the parameters are "learned" in offline. Here we use the road labels generated by K-Means clustering to train a ridge regression model incorporating the camera image and the dynamics of the vehicle.

#### Image Pre-processing
Input images for the ridge regression model were pre-processed with the following methods:

1. Apply Gaussian Blur to smooth the road.
2. Convert to black and white image.
3. Apply principal component anlaysis to attain 99.5% variance.
<div style="display: flex; flex-direction: row; justify-content: space-evenly; width: 100%;"><img src="images/preprocessed_input_image.png" width="60%"></div>

#### Ridge Regression Using Camera Image
<a id="road-segmentation-ridge-1"></a> A naive way to predict the road labels using a regression model is to use the vehicle camera image as input. However, in our experiment, training a ridge regression model based on camera images alone did not generate good predictions. The predicted road labels along with the K-means clustered road labels we used as ground truth for training.

<div style="display: flex; flex-direction: row; justify-content: space-evenly; width: 100%;"><img src="images/without_dynamics3.png" width="80%"></div>

#### Ridge Regression Using Camera Image + Car Dynamics Data
<a id="road-segmentation-ridge-2"></a> 
The dynamics of the vehicle tells more than what the vehicle "sees". We augmented the dimension of the data with the AutoRally vehicle's state and control (longitudinal velocity, lateral velocity, yaw rate, yaw position, global x position, and global y position) and (steering angle, wheel speed) respectively in the ridge regression model. The state and control data have a strong correlation with the environment it operates. The state and control are concatenated together with a pre-processed camera image to formulate the input to the regression model, and the K-Means road labels obtained earlier are used as ground truth labels.

The closed formulation of ridge regression was used to obtain the weight of the ridged regression model. A cross validation with k-fold was run over a series of regularization variables to find a good value. We trained with 50% of the available AutoRally data, and 10-fold was used among the training data set for the cross validation to find a good ridge parameter. The resulting weight is used to predict the road labels with given input consisting of the preprocessed image, state and control.

<div style="display: flex; flex-direction: row; justify-content: space-evenly; width: 100%;"><img src="images/predicted_with_dynamics2.png" width="80%"></div>

The predicted road segmentations are overlayed on top of the input image, and the predicted road maps alone are plotted next to the ground truth obtained with the unsupervised learning. 


## Supervised Turn Classification

<a id="supervised-turn-classification"></a>

### Ground Truth Steering Labels Generation

<a id="supervised-labels"></a>Steering labels for the training and validation datasets were generated automatically using the vehicle states information accompanying the images. We filtered the vehicle's yaw velocity and then thresholded it to fall into one of three classes: right turns, straights, or left turns. The thresholded yaw velocity at the time each image was captured was used to provide the image labels for supervised learning. The classifications along with the measured yaw orientation are displayed below.

<div style="display: flex; flex-direction: row; justify-content: space-evenly; align-items: center; width: 100%;">
    <img src="images/yaw_explanation.png" width="30%" height="30%">
    <img src="images/labeling.png" width="50%">
</div>

### Decision Tree Classification

<a id="supervised-dt"></a> We started by using decision trees to conduct image classification on the segmented road map images. The decision tree recognizes every pixel as a feature for its tree generation process. This has turned into one of the main reasons that deciisions trees may not perform well in image classification problems. The library used was sklearn where tree has min_samples_split as a parameter to determine the minimum number of samples to split a leaf. The value assigned here is min_samples_split = 30, after trying multiple values. The output from this tree is a label (straight, right, or left).

#### Results

The decision tree achieved only a 37.15% accuracy, which is not good. Thus, the team has decided to move with more sophisticated classifiers such as Neural Networks to achieve better results. The figure below shows a random a sample of the results.

<div style="display: flex; flex-direction: row; justify-content: space-evenly; width: 100%;"><img src="images/Decision_tree_turn_classifier2.png" width="80%"></div>

### Neural Network Architecture

<a id="supervised-neural-net"></a>We chose to use a neural network to perform image classification on the segmented road map images. Since we have already generated our feature information in the form of the segmented roadmap, our neural network only needs to map the spatial information to the three possible labels (left turn, straight, right turn). Therefore we elected to use a series of fully connected layers to map the segmented pixel locations to the three categories. We also used one max pooling layer to reduce the image dimensionality and the number of learnable parameters in our network to speed up training. In the figure below, all intermediate fully connected layers are followed by a ReLU activation function. We also show the input and output dimensions for each layer.

<div style="display: flex; flex-direction: row; justify-content: space-evenly; width: 100%;"><img src="images/image_class_net.png" width="17%"></div>

The output class scores are then fed into a softmax function for training which assigns a probability of the image belonging to each of the three classes based on the relative weight of the scores. This allows the gradient to be backpropagated through all three output channels. For validation, we simply take the argmax of the three output channels to determine the most likely class the image belongs to.

#### Results
##### Training and Validation Accuracy

We trained the network for 20 epochs using the Adam optimizer using a cross entropy loss function. The cross entropy loss incorporates softmax which assigns a probability of the image belonging to each class based on the relative scores of the three output channels. The optimier then adjusts the layer weights to maximize the relative scores of the correct classes. We reserved one third of our dataset for validation to prevent overfitting, and plotted the training and validation accuracies for each epoch. We reach about 96% accuracy on the training data and 87% on the validation set.

<div style="display: flex; flex-direction: row; justify-content: space-evenly; width: 100%;"><img src="images/training_acc1.png" width="80%"></div>

##### Validation Examples

We demonstrate the classification results on a sample of validation images. Although the network takes the segmented images as the inputs, we display the raw, unsegmented images for easy visual inspection. The predicted and ground truth labels are displayed for each image. There is one failure case in these examples where a straight was incorrectly labeled as a right turn.

<div style="display: flex; flex-direction: row; justify-content: space-evenly; width: 100%;"><img src="images/image_examples4.png" width="100%"></div>

## Conclusion and Future Work
<a id="conclusion"></a>Our supervised and unsupervised models were able to generate accurate road maps for over 90% of the data, but we realize they may be biased towards specific lighting conditions and road textures. In future work, we'd recommend getting more image data taken in different times of day, weather, and roads.

A decision tree didn't work well for classifying the recommended steering direction based on the road maps, with an accuracy of only 37.15%. We had an easier time getting accurate data using a neural network containing a max pooling layer and four fully connected layers. The neural network performed with 87% validation accuracy on the validation dataset. Neural networks work well with spatial data, so we expected this result.

## Group Member Contributions / Links to Project Code

<a id="contributions"></a>

* Abdulaziz Qwbaiban
    * [Decision Tree Classifier](https://github.com/mlsu2020/DecisionTreeTurnClassifier)
* Jacob Knaup
    * Prepare dataset images and state data
    * Generate labels for training neural network
    * [Implement classification neural network in PyTorch](https://github.com/mlsu2020/turn_classification)
    * Evaluate neural network performance and prepare write-up
* Jaein Lim
    * [Classified road maps using regularized learning](https://github.com/mlsu2020/road-prediction)
* Phong Tran
    * Video and project page editing
    * [Generated road maps using DBSCAN and K-Means](https://github.com/mlsu2020/road-detection)


## References.

<a id="references"></a>

1. Georgia Tech AutoRally Platform. <a id="ref1" href="https://autorally.github.io/">https://autorally.github.io/</a>
2. Determination of Optimal Epsilon Value on DBSCAN Algorithm to Clustering Data on Peatland Hotspots in Sumatra <a id="ref2" href="https://iopscience.iop.org/article/10.1088/1755-1315/31/1/012012/pdf">https://iopscience.iop.org/article/10.1088/1755-1315/31/1/012012/pdf</a>
