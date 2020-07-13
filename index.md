# Road Segmentation and Turn Classification Using Supervised and Unsupervised Learning

## Contents

## Introduction
Autonomous vehicles need to know if they should be turn left, right, or go straight. The goal of our project was to create a machine learning model that can make these classifications based on road images.

## Dataset
Our dataset consisted of images from Georgia Tech's [AutoRally Project](https://autorally.github.io/), a platform for self-driving vehicle research. Robotic cars from this project record the area in front of them while driving around a dirt track.

<div style="display: flex; flex-direction: row; justify-content: space-evenly; width: 100%;">
  <img src="images/img47.png" width="40%">
  <img src="images/img676.png" width="40%">
</div>

## Methods
Our project took the following approach to classifying whether the autorally car should turn left, right, or go straight:

1. Generate a reduced representation of the road scene by segmenting the road from the image. A car's traversable space is limited by the shape of the road they're on, so we figured the **road map** would provide clear training data for a neural network. <div style="display: flex; flex-direction: row; justify-content: space-evenly; width: 100%;"><img src="images/img47.png" width="40%"><img src="images/img47_o.png" width="40%"></div>
2. <div style="color: red; display: inline">Use dynamics data to label each image as going left, right, or straight. @Jacob can you expand on this? Either here or in your section.</div>
3. <div style="color: red; display: inline">Train a neural network on the labelled images.</div>


## Road Segmentation
Road detection can be a difficult problem since the appearance of roads vary depending on lighting conditions, road texture, and the presence of obstacles. The AutoRally data provided us a simplified scene to work with: the images were taken during daytime on a dirt road with no obstacles.

We looked to two clustering algorithms to help us segment the road from the image: DBSCAN and K-Means clustering

### DBSCAN Approach
We started by trying the [DBSCAN clustering algorithm](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html) on the road images. One of the advantages of DBSCAN is that it can determine the number of clusters on its own. This could be helpful in generalized road detection algorithms since we can't always expect there to be consistent lighting conditions.

This is the overall algorithm we used:
1. Pre-processing
    * Crop the image to its bottom 2/3rds to ignore the sky and focus on the road.
    * Apply histogram equalization to make the road stand out from the road edges and car hood.
    * Blur the image to even-out the rocky appearance of the road

<div style="display: flex; flex-direction: row; justify-content: space-evenly; width: 100%;">
<img src="images/dbscan_preprocessing.png" width="80%">
</div>

2. Run DBSCAN to get cluster labels. We used the 3rd largest distance from a [K-NN](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html) search for DBSCAN's epsilon parameter.<sup><a href="#ref1">[1]</a></sup>

<div style="display: flex; flex-direction: row; justify-content: space-evenly; width: 100%;">
<img src="images/dbscan_output.png" width="60%">
</div>

3. Use connected component analysis and a 2D Gaussian to select the cluster that has many points in the center of the image, where we assume the road to be.

<div style="display: flex; flex-direction: row; justify-content: space-evenly; width: 100%;">
<img src="images/dbscan_cc.png" width="95%">
</div>

### Results
Despite our pre-processing steps and DBSCAN parameter tuning, we found the clusterings produced by DBSCAN to still be too sensitive to road aberrations like grass and rocks. 

## Road Segmentation Using K-Means


## References.
1. <a id="ref1" href="https://iopscience.iop.org/article/10.1088/1755-1315/31/1/012012/pdf" style="display: inline">Determination of Optimal Epsilon Value on
DBSCAN Algorithm to Clustering Data on
Peatland Hotspots in Sumatra</a>