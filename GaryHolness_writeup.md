## Writeup Template
### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

** Gary Holness **
**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* I applied a color transformation to YCrCb colorspace
* I also employ binned color features in conjunction with color histograms
* Features were normalized and randomized trianing and testing sets were created.
* Using a sliding window technique, I ran my trained SVM classifier on image patches
  obtaines from sampled image patches returned by the sliding window. This was my implementation 
  of vehicle detection from imagery.
* I build a heatmap for the search windows and label it gray/white colors.
* The heatmaps are averaged over a window of 8 video frames.
* The processing stages were tested individually and the final result was integrated into a single
  pipeline reoutine that performs all of the vehicle detection processing on a single image. This
  routine was created to be called on video frame images.
* I ran my pipeline on the test video frame which was smaller in order to tune parameters
  and evaluate performance as part of the development process.
* I ran my pipeline on the project video

[//]: # (Image References)
[image1]: ./output_images/car_noncar_hog_example.jpg
[image2]: ./output_images/drawimages_hog_test1.jpg
[image3]: ./output_images/efficient_1_5_hog_window_samp.jpg
[image4]: ./output_images/efficient_2_0_hog_window_samp.jpg
[image5]: ./output_images/efficient_2_5_hog_window_samp.jpg
[image6]: ./output_images/efficient_3_0_hog_window_samp.jpg
[image7]: ./output_images/heatmap_hog_test1.jpg
[image8]: ./output_images/heatmapthresh_hog_test1.jpg
[image9]: ./output_images/labelimages_hog_test1.jpg
[video1]: ./test.mp4
[video2]: ./project_result.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
## Writeup / README

My code is implemented in a single file  `vehicle_detector.py`


### Histogram of Oriented Gradients (HOG)

#### 1. HoG

The code for computing HoG features is contained in `find_cars()` (lines 796-873) using a service
routine `get_hog_features()` (lines 105-124).  The HoG features are performed on all color
channels of the selected color space representation (YCrCb).

A test was created for this to visualize the HoG features for a car and non-car image
![hog features][image1]

As can be seen, the different curvatures represented in the shape of a car represent 
a more uniform distribution over gradient angles than for non-car objects.  Code for the
test is called (lines 533, 541) with visualization turned on.  The final processing pipeline
uses lessons learned from this code to compute a single HoG feature response over the entire
rescaled image and then slide a window across the result in order to perform recognition
(lines 830-873). 

#### 2. Image Data Set

I read in images of vehicles and non-vehicles, split them into training/testing set using a 90%
split for the training set (line 623).  From the training set, I used the `StandardScaler()` to
normalize the data.  Once I computed the scaler, I used the result to transform the test set
(lines 632-637).

An example of one of each of the `vehicle` and `non-vehicle` appears below:

![alt text][image1]

I tinkered with parameters impacting HoG features to see what gave good results and decided
upon `color_space=yCrCb', `orient=12`, spatial_size=(16,16), cell_per_block=2, pix_per_cell=10
(lines 506-518).  

#### 2. Explain how you settled on your final choice of HOG parameters.


luma (brightness), the Cb is blue minus luma (B-Y) and the Cr is red minus
luma (R-Y).  The idea behind using YCbCr is that the shadows and bright spots
are represented in the luma, so we have representation of the fact the video frames
have light and dark repreentations.  The Cb and Cr channels represent the color
in the form of Blue and Red.  Ysing YCbCr is intended to handle the color as well
as ways those colors can be lightened or darkened by shadows and bright spots

The bin orientation refers to both the discussion video as well as a HoG publication
that describes HoG effectiveness as up to 9-bin orientations.  Having more fine grained
bin orientation (i.e. more bins) means identification of more slight variations in
edge information.  This can be useful condidering that car bodies are slanted (curvy
in actuality) thus providing features good at picking out different types of cars.

Pixels per cell pix_per_cell= 10 because I wanted a larger area over which to describe
pixel information.  I call this a larger "capture range" because the gradient information,
if representing a larger area 10 x 10, will encode color/shape information in context. That
is, the relationship between adjacent pixels across a larger extent.  What makes a car
different from non-car is the conjunction of features within a capture range. The idea is
the larger the capture range, the more complex (more parts) the conjunction of features
represented.  This stands to improve discrimination between car and non-car objects in
the images.

hog_channel='ALL'
was chosen because it includes all of the color chanel information.  Representation with
all color channels captures more aspects of appearance (color/shape) based features of
objects in images.

orient=12
I chose to have a bit larger number of orientation 'bins' for HoG features.  The rationale
behind this was to campure the broader range of oriented gradients (edges) in a car image.
Cars are generally curved and therefore contain angle information across a boarder range
of angle values.   Non-Car objects tend to have straight edges and do not have the type
of curvature that cars do.

spatial_size=(16,16)
I decided to keep 16 x 16 because I felt it was somewhat of a compression of the original
image imformation in context.  This include aspects of the foreground objects with some
background.  With this spatial size, I believe suitabley depicts object information in
context, that is with surrounding information.

hist_bins=32
I wrestled with this one going back and forth between larger values (liek 32 bins I used) and
smaller values.  The larger values increase specificity in the value of colors depicted.  This
can yield improved discrimination, but there is an issue concerning noise.  If an image is
noisy, the larger bin number will capture that noise. In contrast, a smaller bin value
(I went as small as 8) results generally in higher counts per bin.  While this reduces
the ability to distinguish more specific colors (sligh variations), it provides a sense
of filtering.  That is, noisy pixels might contribute an increment to a bin count, but
if the bin counts are large to begin with, a spurious addition to the count does not
significantly change the bin count relative to the other bins.  Therefore, smaller bin
counts give a type of robustness to noise.  But, this cannot go too far.  Too small
a bin count makes the representation less descriptive.  This speaks to the typical
tradeoff in Machine Learning between model complexity (increase dimensionality in the
model) , generalization/regularization (applicability to new data versus modeling the noise), and
sample complexity (Hoeffding's inequality and relationship between model complexity,
sample size, and how well in-sample and out-of-sample error coincide).


#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using HoG features (lines 829-833), spatial bin features (line 855) , and color
histogram features (line 856). 

### Sliding Window Search

#### 4. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I performed search using a sliding window.  The window overlap, `xy_overlap=0.25`, was 25%.  The
reason for this was to have a smoother transition between adjacent windows because it is more
likely to catch an image patch containing a car.  While this made search longer, it increased
resolution for what can be recognized in the image, less likely to skip over pixels that
indicate the presence of a car.   I implemented multiple scales using something I call
`do_heat_stack()`.  This function takes an array of scales.  I use multiple scales, namely
1.0, 1.5, 2.0, and 2.5 to perform sliding window search.  I certainly need multiple scales
in order to capture both larger and smaller image patches.


Basic HoG Window Search scale=1.0
![hog search 1.0][image2]

Basic HoG Window Search scale=1.5
![hog search 1.5][image3]

Basic HoG Window Search scale=2.0
![hog search 2.0][image4]

Basic HoG Window Search scale=2.5
![hog search 2.5][image5]

Basic HoG Window Search scale=3.0
![hog search 3][image6]

#### 5. Heatmap

Once I return the rectangles from the sliding window search, I compute a heatmap.  This heatmap
includes rectangles from multiple scales, 1.0,1.5,2.0, 2.5.  Heatmap counts are added
across the multiple scales.  The result is thresholded by a value of `hmap_thresh= 5`.

Orginal heatmap image

![alt text][image7]

Thresholded heatmap image

![alt text][image8]

#### 6. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Beginning with values from the quizzes, I thought about what the result of changing parameter
values might mean.  For example, a larger number of histogram bins increases the feature space
but decreates generalization.  The reason for this, for example, is that fewer bins means
higher bin counts.  With higher bin counts, a spurious pixel adding to a bin does not 
increase the bin's probability mass in any significant way.  This means that higher bincounts
can only improve generalization.

I thought about the images in the test video and felt the YCrCB was covered. My choosing it
was a way to explicitly represent shadow information from the actual color information.

I felt the spatial binning and color hisogram could only help since SVMs do well when
dimensionality is high.

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

[![video result](http://img.youtube.com/vi/_iug_AWepuw/0.jpg)](http://www.youtube.com/watch?v=_iug_AWepuw "video result")


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I did a number of things to reduce false positives.  When classifying car instances, I also
(line 413, 866), I call `svc.decision_function()` and use a confidence trheshold of 0.6 on
the Support Vector Machine classification as a way of ensuring that very strong decisions
(high confidence) were admitted.

I also maintain a window of 8 video frames over which I compute the average of 8 multi-scale
heatmaps. These two things together, classifier confidence and multi-scale thresholded heat map,
seemed to work well.

### Here are six frames and their corresponding heatmaps:

![multi scale heatmap][image7]

### Six frames with thresholded heatmap
![thresholded heatmap][image8]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![labeled heatmap][image9]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

This pipeline took a very long time to get working properly because I found I was
making educated guesses about what the paramter should have neen. Moreover, while my goal
was to have descriptive features, there was a practical limitation because a bigger
feature vector dimensionality, the more costly the training.   An improvement would
have perhaps been to set aside a validation set and use the validation set to
find the right parameter value and combination.
