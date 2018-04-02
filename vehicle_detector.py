####
# Gary Holness
#
# Vehicle Detection Project Submission 
#
# This project performs feature extraction, windowed search, and classification in order
# to implement vehicle detection.  The high level approach implements the following steps.
#
##

import os
import glob
import numpy as np


#####
# Open streaming version of the data and load images for
# the two classes, namely car and non-car.
###


#####
# the names for images of cars
###
base_dir= './vehicles/'

image_types = os.listdir(base_dir)

cars= []

for imtype in image_types:
   cars.extend(glob.glob(base_dir + imtype + '/*'))

print("Number of vehicle images found: ",len(cars))
with open("cars.txt",'w') as f:
   for fn in cars:
      f.write(fn + '\n')

car_example_name = './output_images/car_example.jpg'
car_example= cars[0]
#cv2.imwrite(car_example_name,car_example)


#####
#the names for images of non-cars
###

base_dir= './non-vehicles/'

image_types = os.listdir(base_dir)

notcars = []

for imtype in image_types:
   notcars.extend(glob.glob(base_dir + imtype + '/*'))

print("Number of non-vehicle images found: ", len(notcars))
with open("notcars.txt",'w') as f:
   for fn in notcars:
      f.write(fn + '\n')

notcar_example_name= './output_images/notcar_example.jpg'
notcar_example= notcars[0]
#cv2.imwrite(notcar_example_name,notcar_example)


import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#for scikit-learn <= 0.18 use
#from sklearn.cross_validation import train_test_split


#####
# Color conversion routine
##
def convert_color(img, conv='RGB2YCrCb'):
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    if conv == 'RGB2HSV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    if conv == 'RGB2HLS':
        return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    if conv == 'RGB2YUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YUV)


#####
# Hog feature extraction routine
###
def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  block_norm= 'L2-Hys',
                                  transform_sqrt=False, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       block_norm= 'L2-Hys',
                       transform_sqrt=False, 
                       visualise=vis, feature_vector=feature_vec)
        return features

#####
# Spatial binning, rescales image color chanels and vectorizes it
#
# Downsamples the original image to 32 x 32 effectively perserving
# spatial and color information across rows in downsampled image
###
def bin_spatial(img, size=(32, 32)):
    color1 = cv2.resize(img[:,:,0], size).ravel()
    color2 = cv2.resize(img[:,:,1], size).ravel()
    color3 = cv2.resize(img[:,:,2], size).ravel()
    return np.hstack((color1, color2, color3))
                        
#####
# Compute color histogram for each color channel and vectorize it
#
# each color channel is binned via histogram and resulting histogram
# bin counts are concatenated as a single feature vector
###
def color_hist(img, nbins=32):    #bins_range=(0, 256)
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

#####
#
###
def extract_features1(imgs, cspace='RGB', spatial_size=(32, 32),
                        hist_bins=32, hist_range=(0, 256)):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        # Read in each one by one
        image = mpimg.imread(file)
        # apply color conversion if other than 'RGB'
        if cspace != 'RGB':
            if cspace == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif cspace == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif cspace == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif cspace == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        else: feature_image = np.copy(image)      
        # Apply bin_spatial() to get spatial color features
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        # Apply color_hist() also with a color space option now
        hist_features = color_hist(feature_image, nbins=hist_bins, bins_range=hist_range)
        # Append the new feature vector to the features list
        features.append(np.concatenate((spatial_features, hist_features)))
    # Return list of feature vectors
    return features


#####
# Create features from image consisting of
#    - spatial binning
#    - histogram
#    - hog
#
#  and vectorize them by concatenating
##
def single_img_features(img, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True, hog_vis= True):    
    #1) Define an empty list to receive features
    img_features = []
    #2) Apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else:
        feature_image = np.copy(img)      

    #3) Compute spatial features if flag is set
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        #4) Append features to list
        img_features.append(spatial_features)
        #print("single_img_features: spatial_feat, image_features= ",img_features)

    #5) Compute histogram features if flag is set
    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        #6) Append features to list
        img_features.append(hist_features)
        #print("single_img_features: hist_feat, image_features= ",img_features)

    #7) Compute HOG features if flag is set
    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:,:,channel], 
                                    orient, pix_per_cell, cell_per_block, 
                                    vis=False, feature_vec=True))      
        else:
          if hog_vis == True:
            hog_features, hog_image = hog(img[:,:,hog_channel], orientations=orient, \
                                  pixels_per_cell=(pix_per_cell, pix_per_cell), \
                                  cells_per_block=(cell_per_block, cell_per_block), \
                                  block_norm= 'L2-Hys', \
                                  transform_sqrt=False,  \
                                  visualise=hog_vis, feature_vector=True)
          else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)

        #8) Append features to list
        img_features.append(hog_features)
        #print("single_img_features: hog_feat, image_features= ",img_features)


    if ((hog_feat == True) & (hog_vis ==True)):
       #9a) Return concatenated array of features with hog visualization
       return np.concatenate(img_features), hog_image
    else:
       #9b) Return concatenated array of features only
       return np.concatenate(img_features)

    #9) Return concatenated array of features
    #return np.concatenate(img_features)



def extract_features(imgs, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9,
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
   features = []

   for file in imgs:
      #print("extract_features: image= ",file)
      image= mpimg.imread(file)

      feature_vec = single_img_features(image,color_space,spatial_size,hist_bins,orient, \
                           pix_per_cell, cell_per_block, hog_channel, \
                           spatial_feat, hist_feat, hog_feat=hog_feat, hog_vis=False)

      #print("extract_features: feature_vec= ",feature_vec);
      features.append(feature_vec)

   return features



# Here is your draw_boxes function from the previous exercise
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy
    
    
# Define a function that takes an image,
# start and stop positions in both x and y, 
# window size (x and y dimensions),  
# and overlap fraction (for both x and y)
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):

    DEBUG_SLIDE_WINDOW= False

    # If x and/or y start/stop positions not defined, set to image size
    # Compute the span of the region to be searched    
    # Compute the number of pixels per step in x/y
    # Compute the number of windows in x/y
    # Initialize a list to append window positions to
    img_x = img.shape[1]
    img_y = img.shape[0]
  
    if DEBUG_SLIDE_WINDOW: 
      print("img_x= ",img_x)
      print("img_y= ",img_y)
    
    xStep = np.round(xy_window[0] * xy_overlap[0])
    yStep = np.round(xy_window[1] * xy_overlap[1])

    if DEBUG_SLIDE_WINDOW:
      print("xStep= ",xStep)
      print("yStep= ",yStep)
    

    if x_start_stop[0] == None:
        xstart= 0
        xstop = img.shape[1]
    else:
        xstart= x_start_stop[0]
        xstop= x_start_stop[1]
        
    if y_start_stop[0] == None:
        ystart= 0
        ystop= img.shape[0]
    else:
        ystart= y_start_stop[0]
        ystop= y_start_stop[1]

    
    nWindowsX=  np.int_(np.floor(((xstop-xstart - xStep))/xStep))
    nWindowsY=  np.int_(np.floor(((ystop-ystart- yStep))/yStep))

    if DEBUG_SLIDE_WINDOW:
      print("nWindowsX= ",nWindowsX)
      print("nWindowsY= ",nWindowsY)
    
    window_list = []
    # Loop through finding x and y window positions
    #     Note: you could vectorize this step, but in practice
    #     you'll be considering windows one by one with your
    #     classifier, so looping makes sense
        # Calculate each window position
        # Append window position to list
    # Return the list of windows
    for ypos in range(nWindowsY):
        for xpos in range(nWindowsX):
            xcoord= np.int_(xpos*xStep) + xstart
            ycoord= np.int_(ypos*yStep) + ystart
            
            xstop= xcoord + xy_window[0] 
            ystop= ycoord + xy_window[1]
            
            window_pos = ((xcoord, ycoord), (xstop, ystop))
            window_list.append(window_pos)
   
    if DEBUG_SLIDE_WINDOW: 
      print(window_list)

    return window_list

#windows = slide_window(image, x_start_stop=[None, None], y_start_stop=[None, None], 
#                    xy_window=(128, 128), xy_overlap=(0.5, 0.5))
#                       
#window_img = draw_boxes(image, windows, color=(0, 0, 255), thick=6)                    
#plt.imshow(window_img)





# Define a function you will pass an image 
# and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows, clf, scaler, color_space='RGB', xy_window= (64,64), \
                    spatial_size=(32, 32), hist_bins=32, \
                    hist_range=(0, 256), orient=9, \
                    pix_per_cell=8, cell_per_block=2, \
                    hog_channel=0, spatial_feat=True, \
                    hist_feat=True, hog_feat=True):

    #1) Create an empty list to receive positive detection windows
    on_windows = []
    #2) Iterate over all windows in the list
    for window in windows:
        #3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], xy_window)      
        #4) Extract features for that window using single_img_features()
        features = single_img_features(test_img, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat,hog_vis= False)
        #5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        #6) Predict using your classifier
        prediction = clf.predict(test_features)
        confidence = clf.decision_function(test_features)
        #print("prediction= ",prediction, " confidence= ",confidence)
        #7) If positive (prediction == 1) then save the window
        if ((prediction == 1) & (confidence >= 0.6)):
            on_windows.append(window)
    #8) Return windows for positive detections
    return on_windows


#####
# display images tiled in subplot
##
def visualize(fig, rows, cols, imgs, titles,cmap):
   for i, img in enumerate(imgs):
      plt.subplot(rows, cols, i+1)
      plt.title(i+1)
      img_dims = len(img.shape)
      if (img_dims < 3):
         plt.imshow(img,cmap)
         plt.title(titles[i])
      else:
         plt.imshow(img)
         plt.title(titles[i])

   plt.show(block=False)
   plt.pause(3)

car_index= np.random.randint(0,len(cars))
notcar_index= np.random.randint(0,len(notcars))

car_image= mpimg.imread(cars[car_index])
notcar_image= mpimg.imread(notcars[notcar_index])


#####
# Parameters used for feature extraction and classification
#
# The YCbCr color space represents color as brightness and two color difference
# channels, while RGB represents color as mixtures of Red/Green/Blue.  The Y-part is
# luma (brightness), the Cb is blue minus luma (B-Y) and the Cr is red minus
# luma (R-Y).  The idea behind using YCbCr is that the shadows and bright spots 
# are represented in the luma, so we have representation of the fact the video frames
# have light and dark repreentations.  The Cb and Cr channels represent the color
# in the form of Blue and Red.  Ysing YCbCr is intended to handle the color as well
# as ways those colors can be lightened or darkened by shadows and bright spots
#
# The bin orientation refers to both the discussion video as well as a HoG publication
# that describes HoG effectiveness as up to 9-bin orientations.  Having more fine grained
# bin orientation (i.e. more bins) means identification of more slight variations in 
# edge information.  This can be useful condidering that car bodies are slanted (curvy
# in actuality) thus providing features good at picking out different types of cars.
#
# Pixels per cell pix_per_cell= 10 because I wanted a larger area over which to describe
# pixel information.  I call this a larger "capture range" because the gradient information,
# if representing a larger area 10 x 10, will encode color/shape information in context. That
# is, the relationship between adjacent pixels across a larger extent.  What makes a car
# different from non-car is the conjunction of features within a capture range. The idea is
# the larger the capture range, the more complex (more parts) the conjunction of features
# represented.  This stands to improve discrimination between car and non-car objects in
# the images.
#
# hog_channel='ALL'
# was chosen because it includes all of the color chanel information.  Representation with
# all color channels captures more aspects of appearance (color/shape) based features of
# objects in images.
#
# orient=12
# I chose to have a bit larger number of orientation 'bins' for HoG features.  The rationale
# behind this was to campure the broader range of oriented gradients (edges) in a car image.
# Cars are generally curved and therefore contain angle information across a boarder range
# of angle values.   Non-Car objects tend to have straight edges and do not have the type
# of curvature that cars do.
#
# spatial_size=(16,16)
# I decided to keep 16 x 16 because I felt it was somewhat of a compression of the original
# image imformation in context.  This include aspects of the foreground objects with some
# background.  With this spatial size, I believe suitabley depicts object information in
# context, that is with surrounding information.
#
# hist_bins=32
# I wrestled with this one going back and forth between larger values (liek 32 bins I used) and
# smaller values.  The larger values increase specificity in the value of colors depicted.  This
# can yield improved discrimination, but there is an issue concerning noise.  If an image is
# noisy, the larger bin number will capture that noise. In contrast, a smaller bin value
# (I went as small as 8) results generally in higher counts per bin.  While this reduces
# the ability to distinguish more specific colors (sligh variations), it provides a sense
# of filtering.  That is, noisy pixels might contribute an increment to a bin count, but
# if the bin counts are large to begin with, a spurious addition to the count does not
# significantly change the bin count relative to the other bins.  Therefore, smaller bin
# counts give a type of robustness to noise.  But, this cannot go too far.  Too small 
# a bin count makes the representation less descriptive.  This speaks to the typical
# tradeoff in Machine Learning between model complexity (increase dimensionality in the
# model) , generalization/regularization (applicability to new data versus modeling the noise), and
# sample complexity (Hoeffding's inequality and relationship between model complexity,
# sample size, and how well in-sample and out-of-sample error coincide).

color_space= 'YCrCb'   #Can be RGB, HSV, LUV, HLS, YUV, YCrCb
#color_space= 'RGB'   #Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient= 12  #12
pix_per_cell = 10  #10 8
cell_per_block= 2  #2
hog_channel = 'ALL'  #0
spatial_size= (16,16)  #(32,32)
hist_bins= 32  #32
hist_range=(40, 256)  #(0,256)
spatial_feat = True
hist_feat= True
hog_feat = True
hog_vis= True

DO_VIS = False

if DO_VIS:
  #####
  # show example car and non-car along with their hog feature vector visualizations
  # 
  # To do this, I turn on hog_vis so that I can get an image of the hog features.
  #
  # This presented me a problem while implementing because (unlike Java), Python is wierd
  # in that it allows the same function to have different versions of return type.  The
  # same function can return a single return type or a multiple return. Very wierd! At
  # least to me.
  ###
  car_features, car_hog_image = single_img_features(car_image, color_space= color_space, \
                                                spatial_size= spatial_size, hist_bins= hist_bins, \
                                                orient= orient, pix_per_cell= pix_per_cell, \
                                                cell_per_block= cell_per_block, hog_channel= 0, \
                                                spatial_feat= spatial_feat, hist_feat= hist_feat, \
                                                hog_feat= hog_feat, hog_vis= True)                                      

 
  notcar_features, notcar_hog_image = single_img_features(notcar_image, color_space= color_space, \
                                                spatial_size= spatial_size, hist_bins= hist_bins, \
                                                orient= orient, pix_per_cell= pix_per_cell, \
                                                cell_per_block= cell_per_block, hog_channel= 0, \
                                                spatial_feat= spatial_feat, hist_feat= hist_feat, \
                                                hog_feat= hog_feat, hog_vis= True)


  images= [car_image, car_hog_image, notcar_image, notcar_hog_image]
  titles= ['car image', 'car HoG image', 'not car image', 'not car HoG image']
  fig = plt.figure(figsize=(12,13))
  visualize(fig,1,4,images, titles,cmap='hot')
  fig.savefig('./output_images/car_noncar_hog_example.jpg')




#####
# Load the image data and extract feature vectors from them
###

print("Computing features...")

t= time.time()

RANDOM_SUBSAMPLE_DATASET= False

if RANDOM_SUBSAMPLE_DATASET:
  n_samples= 500

  random_idxs = np.random.randint(0,len(cars), n_samples)

  test_cars= np.array(cars)[random_idxs]
  test_notcars = np.array(notcars)[random_idxs]

else:
  test_cars= cars
  test_notcars = notcars


#####
# This version, extract_features, calls the single_img_features routine in a loop
# in order to load the entire data set for car and non-car examples
###
car_features= extract_features(test_cars, color_space=color_space, spatial_size= spatial_size, \
                        hist_bins= hist_bins, orient= orient, \
                        pix_per_cell= pix_per_cell, cell_per_block= cell_per_block, hog_channel= hog_channel, \
                        spatial_feat= spatial_feat, hist_feat= hist_feat, hog_feat= hog_feat)



notcar_features= extract_features(test_notcars, color_space=color_space, spatial_size= spatial_size, \
                        hist_bins= hist_bins, orient= orient, \
                        pix_per_cell= pix_per_cell, cell_per_block= cell_per_block, hog_channel= hog_channel, \
                        spatial_feat= spatial_feat, hist_feat= hist_feat, hog_feat= hog_feat)

print("car_features len= ",len(car_features))
print("car_features[0] len= ",len(car_features[0]))
print("notcar_features len= ",len(notcar_features))

print(time.time() - t, "Seconds to compute features...")

#####
# Create data set by putting together
# feature vectors for cars on top of feature vectors for notcars
###

X= np.vstack((car_features, notcar_features)).astype(np.float64)

#Create Labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))


####
# partition into trandomized training and testing sets
#
# Note:  no data snooping, so make sure scaler is created
#        on the training data and applied to both training/testing data
#       
##
rand_state= np.random.randint(0,100)

X_train_unscaled, X_test_unscaled, y_train, y_test = train_test_split(X,y,test_size=0.1,random_state=rand_state)

print("X_train_unscaled len= ",len(X_train_unscaled))
print("X_test_unscaled len= ",len(X_test_unscaled))
print("y_train len= ",len(y_train))
print("y_test len= ",len(y_test))

#calculate scaler from training set

X_scaler = StandardScaler().fit(X_train_unscaled)

#apply scaler to training set and test set
X_train= X_scaler.transform(X_train_unscaled)

X_test= X_scaler.transform(X_test_unscaled)

print("orientations= ",orient)
print("pixels per cell= ",pix_per_cell)
print("cells per block= ",cell_per_block)
print("histogram bins= ",hist_bins)
print("spatial size= ",spatial_size)
print("Feature vector dimensionality= ",len(X_train[0]))


#####
# Linear Support Vector Machine Classifier
##

svc = LinearSVC()

print("Training SVC...")
t= time.time()
svc.fit(X_train,y_train)

print(time.time() -t, "Seconds to train SVC")

#####
# SVC performance on test data
###

accuracy = round(svc.score(X_test, y_test),4)
print("SVC test accuracy= ",accuracy)


#####
# set bounds on vertical span
# for search windows in image
#
# set overlap for search windows.
# assumes horizontal and vertical directions.
###

IMG_ROWS= 720
IMG_COLS= 1280
HOOD_HEIGHT=64

#y_start_stop= [400, 720-HOOD_HEIGHT]
#y_start_stop= [368, 720-HOOD_HEIGHT] 
y_start_stop= [240, 720-HOOD_HEIGHT] 
#y_start_stop= [304, 720-HOOD_HEIGHT]
x_start_stop= [None, None]
xoverlap= 0.25
yoverlap= 0.25
xy_overlap= (xoverlap, yoverlap)

xwindow= 64
ywindow= 64
xy_window= (xwindow, ywindow)

clf = svc
scaler= X_scaler

#####
# perform the sliding window algorithm to tesselate the input image and
# return the windowed image samples where a car was found
###
def do_slide(img, scaler= scaler, x_start_stop= x_start_stop, y_start_stop= y_start_stop, \
                              xy_window= xy_window, xy_overlap= xy_overlap, clf= clf, \
                              spatial_size= spatial_size, hist_bins= hist_bins, hist_range= hist_range,\
                              orient= orient, cell_per_block= cell_per_block, \
                              pix_per_cell= pix_per_cell, hog_channel= hog_channel):

    #####
    # Get search windows
    ###
    windows= slide_window(img, x_start_stop= x_start_stop, y_start_stop= y_start_stop, \
                          xy_window= xy_window, xy_overlap=xy_overlap)

    #####
    # Record hits (presence of cars)
    ###
    hit_windows= search_windows(img, windows, clf, scaler, color_space= color_space, xy_window= xy_window, \
                    spatial_size= spatial_size, hist_bins= hist_bins, \
                    hist_range= hist_range, orient= orient, \
                    pix_per_cell= pix_per_cell, cell_per_block= cell_per_block, \
                    hog_channel=hog_channel, spatial_feat=True, \
                    hist_feat=True, hog_feat=True)

    window_img = draw_boxes(draw_img, hit_windows, color=(0,0,255), thick=6)

    print(time.time() - t1,"seconds to search one image, resulting in ",len(windows),"windows")

    return window_img
     

#####
# Test out the trained classifier
###

DO_SLIDING_WINDOW_TEST= False

searchpath = './test_images/*.jpg'
example_images= glob.glob(searchpath)

images= []
titles= []

test_images= []

i=0

#####
# Load the test images here because
# it is reused in two places
###
for img_src in example_images:
  img= mpimg.imread(img_src)
  print("img.size= ",img.shape)
  test_images.append(img)
  titles.append('test' + str(i+1) + '.jpg')
  i+= 1

   
  
if DO_SLIDING_WINDOW_TEST:
  i= 0
  for img in test_images:
     print("sliding windows test, finding windows in ", titles[i])
     t1 = time.time()

     draw_img = np.copy(img)

     #scale image intensities to closed interval [0,1] in Real number line
     #to match the representation of PNG images [0,1] used for training
     img= img.astype(np.float32)/255

     #print(np.min(img),np.max(img))

     the_image= do_slide(img, scaler= scaler, x_start_stop= x_start_stop, y_start_stop= y_start_stop, \
                 xy_window= xy_window, xy_overlap= xy_overlap, clf= clf, \
                 spatial_size= spatial_size, hist_bins= hist_bins, hist_range= hist_range, \
                 orient= orient, cell_per_block= cell_per_block, \
		 pix_per_cell= pix_per_cell, hog_channel= hog_channel)

     print(time.time() -t, "Seconds to find windows")

     images.append(the_image)
     #titles.append('')
     i+= 1
   
  #fig = plt.figure(figsize=(12,18), dpi=300)
  fig = plt.figure(figsize=(6,9), dpi=300)
  visualize(fig, 4, 3, images, titles,cmap='hot')
  fig.savefig('./output_images/test_vehicle_detection.jpg')

  


#####
# Find cars by computing HoG features for entire image and subsampling through
# sliding window and classifying.
###
# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, \
              hog_channel, spatial_size, hist_bins, conv):
    draw_img = np.copy(img)
    
    the_rectangles= []

    img = img.astype(np.float32)/255
    
    img_tosearch = img[ystart:ystop,:,:]

    #ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
    ctrans_tosearch = convert_color(img_tosearch, conv=conv)

    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1 
    nfeat_per_block = orient*cell_per_block**2
    
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1
    
    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 

            if (hog_channel == 'ALL'):
               hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
               hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
               hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
            else:
               hog_feature = hog_feat1

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
          
            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))    
            #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
            test_prediction = svc.predict(test_features)
            confidence = svc.decision_function(test_features)
            #print("test_prediction= ",test_prediction, " confidence= ",confidence)
     
            
            if ((test_prediction == 1) & (confidence >= 0.6)):
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                the_rectangles.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)))
                cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6) 
                
    return the_rectangles, draw_img


def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes
    
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

#####
# draw boxes using labels (gray color combo of recangles)
##
def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img 



#####
# Test version of car finder that uses HoG on entire image followed
# by subsampling 
###

DO_TEST_HOG_SCALES= False
conv='RGB2YCrCb'
#conv='RGB2YUV'

if DO_TEST_HOG_SCALES:
   #####
   # Test HoG windows for scale=1.0
   ###
   scale= 1.0

   rect_images= []
   i= 0

   for img in test_images: 
      t= time.time()
      rectangles, rect_image = find_cars(img, y_start_stop[0], y_start_stop[1], scale, svc, X_scaler, \
                                       orient, pix_per_cell, cell_per_block, \
                                       hog_channel, spatial_size, hist_bins, conv)
      rect_images.append(rect_image)

      print(time.time() - t," seconds to find HoG windows in ",titles[i])
      i+= 1


   fig = plt.figure(figsize=(12,13))
   visualize(fig,4,3,rect_images, titles,cmap='hot')
   fig.savefig('./output_images/efficient_hog_window_samp.jpg')


   #####
   # Test HoG windows for scale=1.5
   ###
   rect_images= []
   scale=1.5
   i= 0

   for img in test_images:
      t= time.time()
      rectangles, rect_image = find_cars(img, y_start_stop[0], y_start_stop[1], scale, svc, X_scaler, \
                                       orient, pix_per_cell, cell_per_block, \
                                       hog_channel, spatial_size, hist_bins, conv)
      rect_images.append(rect_image)

      print(time.time() - t," seconds to find HoG windows in ",titles[i])
      i+= 1


   fig = plt.figure(figsize=(12,13))
   visualize(fig,4,3,rect_images, titles,cmap='hot')
   fig.savefig('./output_images/efficient_1_5_hog_window_samp.jpg')


   #####
   # Test HoG windows for scale=2.5
   ###
   rect_images= []
   scale=2.0
   i= 0

   for img in test_images:
      t= time.time()
      rectangles, rect_image = find_cars(img, y_start_stop[0], y_start_stop[1], scale, svc, X_scaler, \
                                       orient, pix_per_cell, cell_per_block, \
                                       hog_channel, spatial_size, hist_bins, conv)
      rect_images.append(rect_image)

      print(time.time() - t," seconds to find HoG windows in ",titles[i])
      i+= 1


   fig = plt.figure(figsize=(12,13))
   visualize(fig,4,3,rect_images, titles,cmap='hot')
   fig.savefig('./output_images/efficient_2_0_hog_window_samp.jpg')


   #####
   # Test HoG windows for scale=2.5
   ###
   rect_images= []
   scale=2.5
   i= 0

   for img in test_images:
      t= time.time()
      rectangles, rect_image = find_cars(img, y_start_stop[0], y_start_stop[1], scale, svc, X_scaler, \
                                       orient, pix_per_cell, cell_per_block, \
                                       hog_channel, spatial_size, hist_bins, conv)
      rect_images.append(rect_image)

      print(time.time() - t," seconds to find HoG windows in ",titles[i])
      i+= 1


   fig = plt.figure(figsize=(12,13))
   visualize(fig,4,3,rect_images, titles,cmap='hot')
   fig.savefig('./output_images/efficient_2_5_hog_window_samp.jpg')


   #####
   # Test HoG windows for scale=3.0
   ###
   rect_images= []
   scale=3.0
   i= 0

   for img in test_images:
      t= time.time()
      rectangles, rect_image = find_cars(img, y_start_stop[0], y_start_stop[1], scale, svc, X_scaler, \
                                       orient, pix_per_cell, cell_per_block, \
                                       hog_channel, spatial_size, hist_bins, conv)
      rect_images.append(rect_image)

      print(time.time() - t," seconds to find HoG windows in ",titles[i])
      i+= 1

   fig = plt.figure(figsize=(12,13))
   visualize(fig,4,3,rect_images, titles,cmap='hot')
   fig.savefig('./output_images/efficient_3_0_hog_window_samp.jpg')



#####
# perform heatmap using stack of different scales
##
def do_heat_stack(scales, img, ystart, ystop, svc, X_scaler, orient, pix_per_cell, cell_per_block, \
              hog_channel, spatial_size, hist_bins, conv):

   heatmap_img = np.zeros_like(img[:,:,0])

   for scale in scales:
      rects, rect_img = find_cars(img, ystart, ystop, scale, svc, X_scaler, \
                                 orient, pix_per_cell, cell_per_block, \
                                 hog_channel, spatial_size, hist_bins, conv)

      heatmap_img= add_heat(heatmap_img, rects)    

   return heatmap_img
   

#####
# Test out heat map
###

img= test_images[0]

heatmap_images = []
heatmap_titles = titles

scales= [1.0, 1.5, 2.0, 2.5]

DO_HEAT_MAP_TEST= False

from scipy.ndimage.measurements import label

if DO_HEAT_MAP_TEST:
  i= 0

  for img in test_images:
    heatmap_img= do_heat_stack(scales, img, y_start_stop[0], y_start_stop[1], svc, X_scaler, \
                             orient, pix_per_cell, cell_per_block, \
                             hog_channel, spatial_size, hist_bins,conv)

    #heatmap_img = heatmap_img/len(scales)

    heatmap_images.append(heatmap_img)

    i+= 1

  fig = plt.figure(figsize=(12,13))
  visualize(fig,4,3,heatmap_images, heatmap_titles,cmap='hot')
  fig.savefig('./output_images/heatmap_hog_test1.jpg')



  #####
  # Test out thresholded heatmap
  ###
  heatmap_images_thresh= []

  hmap_thresh= 5

  for heatmap_img in heatmap_images:
    heatmap_img_thresh= apply_threshold(heatmap_img, hmap_thresh)
    heatmap_images_thresh.append(heatmap_img_thresh)

  fig = plt.figure(figsize=(12,13))
  visualize(fig,4,3,heatmap_images_thresh, heatmap_titles,cmap='hot')
  fig.savefig('./output_images/heatmapthresh_hog_test1.jpg')



  #####
  # Test the labeled thresholded heatmap
  ##
  label_images = []
  the_labels = []
  for heatmap_img in heatmap_images_thresh:
     labels = label(heatmap_img)
     label_images.append(labels[0])
     the_labels.append(labels)

  fig = plt.figure(figsize=(12,13))
  visualize(fig,4,3,label_images, heatmap_titles,cmap='gray')
  fig.savefig('./output_images/labelimages_hog_test1.jpg')


  #####
  # Test draw_labeled_bboxes
  ###

  draw_images= []
  num_images= len(test_images)

  for i in range(0,num_images):
     img= test_images[i]
     labels= the_labels[i]
   
     draw_img = draw_labeled_bboxes(img, labels)
     draw_images.append(draw_img)

  fig = plt.figure(figsize=(12,13))
  visualize(fig,4,3,draw_images, heatmap_titles,cmap='gray')
  fig.savefig('./output_images/drawimages_hog_test1.jpg')



#####
# final processing pipeline
###
from queue import Queue

hmap_window_size= 8

global hmap_win
hmap_win= Queue(hmap_window_size)

def process_image(img):
  hmap_thresh= 5
  heatmap_img= do_heat_stack(scales, img, y_start_stop[0], y_start_stop[1], svc, X_scaler, \
                             orient, pix_per_cell, cell_per_block, \
                             hog_channel, spatial_size, hist_bins,conv)
  DO_AVERAGE= True

  if (DO_AVERAGE):
    if (hmap_win.full()):
       hmap_win.get() 
       hmap_win.put(heatmap_img)
       num_maps= 8
    else:
       hmap_win.put(heatmap_img)
       #num_maps = hmap_win.qsize()
       num_maps= 8

    #print("hmap_win.qsize()= ",hmap_win.qsize());

    hlist= list(hmap_win.queue)

    avg_heatmap= np.zeros_like(hlist[0],dtype=np.float)

    for hmap in hlist:
      avg_heatmap = avg_heatmap + np.array(hmap,dtype=np.float)/num_maps
    

      avg_heatmap = np.array(np.round(avg_heatmap),dtype=np.uint8) 

    #print(avg_heatmap)

    #heatmap_img_t = apply_threshold(heatmap_img, hmap_thresh)
    #hmap_thresh=5
    heatmap_img_t = apply_threshold(avg_heatmap, hmap_thresh)
  else:
    heatmap_img_t = apply_threshold(heatmap_img, hmap_thresh)

  labels= label(heatmap_img_t)
  
  draw_img = draw_labeled_bboxes(np.copy(img), labels)

  return draw_img
  

from moviepy.editor import VideoFileClip
from IPython.display import HTML

#test_output = 'test.mp4'
#test_video= 'test_video.mp4'

test_output = 'project_result.mp4'
test_video= 'project_video.mp4'

clip = VideoFileClip(test_video)
test_clip = clip.fl_image(process_image)

test_clip.write_videofile(test_output,audio=False)

HTML("""<video width="960" height="540" controls> <source src="{0}"> </video>""".format(test_output))
  
