## Writeup Template
### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

** Gary Holness **

**Vehicle Detection Project**

This is a resubmission that encorporates feedback from a previous review.
I tried a number of additional steps and was able to improve performance.
The track holds longer for the white car.

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
[image10]: ./output_images/efficient_hog_window_samp.jpg
[video1]: ./test.mp4
[video2]: ./project_result.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
## Writeup / README

My code is implemented in a single file  `vehicle_detector.py`
My video is implemented in a single file  `project_result.mp4'


### Histogram of Oriented Gradients (HOG)

#### 1. HoG

The code for computing HoG features is contained in `find_cars()` (lines 813-890) using a service
routine `get_hog_features()` (lines 105-124).  The HoG features are performed on all color
channels of the selected color space representation (YCrCb).

A test was created for this to visualize the HoG features for a car and non-car image
![hog features][image1]
As can be seen, the different curvatures represented in the shape of a car represent 
a more uniform distribution over gradient angles than for non-car objects.  Code for the
test is called (lines 533, 541) with visualization turned on.  The final processing pipeline
uses lessons learned from this code to compute a single HoG feature response over the entire
rescaled image and then slide a window across the result in order to perform recognition
(lines 851-888). 

#### 2. Image Data Set

I read in images of vehicles and non-vehicles, split them into training/testing set using a 90%
split for the training set (line 624).  From the training set, I used the `StandardScaler()` to
normalize the data.  Once I computed the scaler, I used the result to transform the test set
(lines 633-638).

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
For this reason, I used orient=12 (line 508) to have a little bit more fine grained
distinction among edge orientations.

Pixels per cell pix_per_cell= 10 (line 509) because I wanted a larger area over which to describe
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

spatial_size=(32,32)
I decided to use 32 x 32 because I felt it was somewhat of a compression of the original
image imformation in context.  I didn't want to compress too much and experimented
during my development with 16 x 16.  Because the scaling is linear, I felt going too
far with compression may throw out valuable image information.  This include aspects
of the foreground objects with some background.  With this spatial size, I believe
suitabley depicts object information in context, that is with surrounding information.

hist_bins=16
I wrestled with this one going back and forth between larger values (like 32 bins I used) and
smaller values (I experimented as small as 8).  The larger values increase specificity in the
value of colors depicted.  This can yield improved discrimination, but there is an issue
concerning noise.  If an image is noisy, the larger bin number will capture that noise. In
contrast, a smaller bin value (I went as small as 8) results generally in higher counts per
bin.  While this reduces the ability to distinguish more specific colors (sligh variations),
it provides a sense of filtering.  That is, noisy pixels might contribute an increment to a
bin count, but if the bin counts are large to begin with, a spurious addition to the count does not
significantly change the bin count relative to the other bins.  Therefore, smaller bin
counts give a type of robustness to noise.  But, this cannot go too far.  Too small
a bin count makes the representation less descriptive.  This speaks to the typical
tradeoff in Machine Learning between model complexity (increase dimensionality in the
model) , generalization/regularization (applicability to new data versus modeling the noise), and
sample complexity (Hoeffding's inequality and relationship between model complexity,
sample size, and how well in-sample and out-of-sample error coincide).  In all this resulted
in a high dimensional features vector totaling 6720 dimensions.  Fortunately SVM's are great
with high dimensional data.


#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The training set represents 90% of the randomized data (15984 instances) set while the test set
represents the remaining 10% (1776 instances).  A validation set is carved out of the training
set (10% of training set) and is used to select be best value for the regularization paramerter, C,
for a Linear SVM.  I trained a linear SVM (line 665) using HoG features (lines 846-849), spatial
bin features (line 872) , and color histogram features (line 873).  In order to improve the linear
SVM classifier, I used `GridSearchCV` to employ validation using 10% of the training data to find
the best regularization parameter, C, that best fits the model.   The values for C searched by
'GridSearchCV' (line 654) were `C= 0.0001, 0.001, 0.01, 0.1, 1.0, 10, 100, 1000`.  Consistently the
optimal value for C was 0.001.   Training the SVC classifier took 159.68 seconds.  My trained Linear
SVM classifier was tested on a test set resulting in accuracy of 99.04%.  This was quite good.

### Sliding Window Search

#### 4. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I performed search using a sliding window.  The window overlap, `xy_overlap=0.75`, was 75%.  The
reason for this was to have a smooth transition between adjacent windows because it is more
likely to catch an image patch containing a car.  While this made search longer, it increased
resolution for what can be recognized in the image, less likely to skip over pixels that
indicate the presence of a car.   I implemented multiple scales using something I call
`do_heat_stack()`.  This function takes an array of scales.  I use multiple scales, namely
1.0, 1.2, 1.5, and 2.0 to perform sliding window search.  I certainly need multiple scales
in order to capture both larger and smaller image patches.  

During development, I experimented with many scales.  Examples of this for
scales 1.0, 1.5, 2.0, 2.5, and 3.0 appear below. In my experiments, I wanted to have 
a range of scales from small to large to guage the impact on recognition.  I found that
the smaller scales were more effective in detection.  So I included smaller scales along
with a resonably large scale. The small scales are purposed with addressing when the
cars are farther (small)  and the large scale for addressing when cars are close by (larger).

Basic HoG Window Search scale=1.0
![hog search 1.0][image10]

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
includes rectangles from multiple scales, 1.0,1.2,1.5, 2.0.  Heatmap counts are added
across the multiple scales.  The result is thresholded by a value of `hmap_thresh= 4`.

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

[![video result](http://img.youtube.com/vi/bs20QiZR06k/0.jpg)](http://www.youtube.com/watch?v=bs20QiZR06k "video result")


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I did a number of things to reduce false positives.  When classifying car instances
(line 410, 413, 879, 883), I call `svc.decision_function()` and use a confidence trheshold of 0.15 on
the Support Vector Machine classification as a way of ensuring that stronger decisions
(high confidence) were admitted.

I also maintain a window of 8 video frames over which I compute the average of 8 multi-scale
heatmaps (line 1164, 1176-1197). These two things together, classifier confidence and multi-scale
thresholded heat map, seemed to work well.

### Here are six frames and their corresponding heatmaps:

![multi scale heatmap][image7]

### Six frames with thresholded heatmap
![thresholded heatmap][image8]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![labeled heatmap][image9]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image2]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

This pipeline took a very long time to get working properly because I found I was
making educated guesses about what the paramter should have neen. Moreover, while my goal
was to have descriptive features, there was a practical limitation because a bigger
feature vector dimensionality, the more costly the training.   An improvement would
have perhaps been to set aside a validation set and use the validation set to
find the right parameter value and combination.

This is a resubmission representing suggestions received from review of a project submission.
After trying out suggestions for changes to `pix_per_cell` as well as the number of histogram
bins and orientations, I found the performance of my system didn't improve, rather it got worse.
The suggestion of using 'GridSearchCV` improved my linear classifier performance from 98% accuracy
to 99% accuracy.   What worked better for me was to increase the window size of the average heatmap
from 5 to 8, decreasing the confidence threshold using svc.decision_function from 0.6 to 0.15, and
setting `x_start_stop=[200, None]` in order to exclude the left shoulder of the roadway from
consideration for sliding window.   All of my parameters resulted in a processing iteration averaging
between 3.2 secons and 4.2 seconds.  This translated to a processing of the video of roughly an
hour on my old mid-2010 MacPro running OSX 10.11 (El Capitain).  It would have been nice to
have a more modern machine to speed up processing.

In addition one thing I had considered was to augment the data-set with images of cars that were
side views taken of the white car by making snapshots from the video and cropping.  From the
browsing around of the data set, I found that many of the training images were rear views of
cars.  Once including my own snaphots, the plan would have been to boost side-view and 3/4ths views
of the white car by oversampling and saving it to the data set.  I imagine when creating street
scene data-sets, particularly of cars, one would have data that represents 360-degree views perhaps
in 15-degree increments in order to get more image information to train the model in better
recognizing cars.
