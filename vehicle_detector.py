#####
# Gary Holness
#
# Vehicle Detection Project Submission 
#
# This project performs feature extraction, windowed search, and classification in order
# to implement vehicle detection.  The high level approach implements the following steps.
#
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
    # If x and/or y start/stop positions not defined, set to image size
    # Compute the span of the region to be searched    
    # Compute the number of pixels per step in x/y
    # Compute the number of windows in x/y
    # Initialize a list to append window positions to
    img_x = img.shape[1]
    img_y = img.shape[0]
    
    print("img_x= ",img_x)
    print("img_y= ",img_y)
    
    xStep = np.round(xy_window[0] * xy_overlap[0])
    yStep = np.round(xy_window[1] * xy_overlap[1])
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
    
    print(window_list)
    return window_list

#windows = slide_window(image, x_start_stop=[None, None], y_start_stop=[None, None], 
#                    xy_window=(128, 128), xy_overlap=(0.5, 0.5))
#                       
#window_img = draw_boxes(image, windows, color=(0, 0, 255), thick=6)                    
#plt.imshow(window_img)





# Define a function you will pass an image 
# and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows, clf, scaler, color_space='RGB', 
                    spatial_size=(32, 32), hist_bins=32, 
                    hist_range=(0, 256), orient=9, 
                    pix_per_cell=8, cell_per_block=2, 
                    hog_channel=0, spatial_feat=True, 
                    hist_feat=True, hog_feat=True):

    #1) Create an empty list to receive positive detection windows
    on_windows = []
    #2) Iterate over all windows in the list
    for window in windows:
        #3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))      
        #4) Extract features for that window using single_img_features()
        features = single_img_features(test_img, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
        #5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        #6) Predict using your classifier
        prediction = clf.predict(test_features)
        #7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    #8) Return windows for positive detections
    return on_windows



def visualize(fig, rows, cols, imgs, titles):
   for i, img in enumerate(imgs):
      plt.subplot(rows, cols, i+1)
      plt.title(i+1)
      img_dims = len(img.shape)
      if (img_dims < 3):
         plt.imshow(img,cmap='hot')
         plt.title(titles[i])
      else:
         plt.imshow(img)
         plt.title(titles[i])

   plt.show(block=False)
   plt.pause(5)

car_index= np.random.randint(0,len(cars))
notcar_index= np.random.randint(0,len(notcars))

car_image= mpimg.imread(cars[car_index])
notcar_image= mpimg.imread(notcars[notcar_index])


#parameters 

color_space= 'YCrCb'   #Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient= 9
pix_per_cell = 10  #10 8
cell_per_block= 2
hog_channel = 'ALL'  #0
spatial_size= (32,32)  #(16,16)
hist_bins= 32
spatial_feat = True
hist_feat= True
hog_feat = True
hog_vis= True

DO_VIS = True

if DO_VIS:
  #####
  # show example car and non-car along with their hog feature vector visualizations
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
  visualize(fig,1,4,images, titles)




#####
# Load the image data and extract feature vectors from them
###

print("Computing features...")

t= time.time()

RANDOM_SUBSAMPLE_DATASET= False

if RANDOM_SUBSAMPLE_DATASET:
  n_samples= 1000

  random_idxs = np.random.randint(0,len(cars), n_samples)

  test_cars= np.array(cars)[random_idxs]
  test_notcars = np.array(notcars)[random_idxs]

else:
  test_cars= cars
  test_notcars = notcars



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
svc.fit(X_train,y_train)

t= time.time()

print(time.time() -t, "Seconds to train SVC")

#####
# SVC performance on test data
###

accuracy = round(svc.score(X_test, y_test),4)
print("SVC test accuracy= ",accuracy)
