import numpy as np
import cv2
from optparse import OptionParser
import math
import random as rng
######################
# Reads an image, applies a laplacian, perform distance transform and watershed
#  see https://docs.opencv.org/3.4/d2/dbd/tutorial_distance_transform.html
#
# - Laplace transformation, saved as contourdir/example1.jpg
# - applies threshold, saved as contourdir/example2.jpg
# - applies distance transform, saved as contourdir/example3.jpg
# - applies threshold again, saved as contourdir/example4.jpg
# - finds markers, saved as contourdir/example5.jpg
# - performs watershed, saved in contourdir/example6.jpg and example7.jpg
#
# Second elaboration, from original contours
# - finds contours, saved as contourdir/example8.jpg, example8b.jpg
# - find lines, saved as contourdir/example8c.jpg
# - applies distance transform, saved as contourdir/example9.jpg
# - applies threshold to dst transformed, saved as contourdir/example10.jpg
# - finds markers, saved as contourdir/example11.jpg
# - performs watershed, saved in contourdir/example12.jpg and example13.jpg
#
# Parameters
# -I image name in slices directory
# -T threshold value (default = 80



threshold_value = 80   # can be adjusted depending on acutal images

slicedir = "Zslices/"
graydir = "gray"
threshdir = "thresh"
contourdir = "contour/"
parser = OptionParser()
parser.add_option("-I", "--image", dest="fname",metavar="IM_NAME",
                  help="Choose Image Name")
parser.add_option("-T", "--thvalue", dest="threshold_value",metavar="TH_VALUE",
                  help="Choose Threshold value")
(optlist, args) = parser.parse_args()
if not optlist.fname:
	errmsg='This programs needs the image name as a command line argument'
	raise SyntaxError(errmsg)

if not optlist.threshold_value:
	print("Using default threshold value %s ", threshold_value)
else:
	threshold_value=float(optlist.threshold_value)

fname=optlist.fname

print("Original Image "+fname)
src = cv2.imread(slicedir+fname+'.png')
print("Applying Laplace transform (test.jpg)")

kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float32)
# do the laplacian filtering as it is
# well, we need to convert everything in something more deeper then CV_8U
# because the kernel has some negative values,
# and we can expect in general to have a Laplacian image with negative values
# BUT a 8bits unsigned int (the one we are working with) can contain values from 0 to 255
# so the possible negative number will be truncated
imgLaplacian = cv2.filter2D(src, cv2.CV_32F, kernel)
sharp = np.float32(src)
imgResult = sharp - imgLaplacian
# convert back to 8bits gray scale
imgResult = np.clip(imgResult, 0, 255)
imgResult = imgResult.astype('uint8')
imgLaplacian = np.clip(imgLaplacian, 0, 255)
imgLaplacian = np.uint8(imgLaplacian)
#cv.imshow('Laplace Filtered Image', imgLaplacian)
#cv2.imshow('New Sharped Image', imgResult)
cv2.imwrite(contourdir+'test1.jpg', imgResult)

# thresholding
print("Applying threshold (test2.jpg)")

####### non uso la trasformata...
bw = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
_, bw = cv2.threshold(bw, threshold_value, 255, cv2.THRESH_BINARY) # | cv2.THRESH_OTSU)
#bw = cv2.adaptiveThreshold(bw, 255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,17,2)
cv2.imwrite(contourdir+'test2.jpg', bw)


print("Finding contours (test3.jpg)")

contours2, hierarchy2 = cv2.findContours(bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
print(len(contours2))

blank_image2 = np.zeros((src.shape[0], src.shape[1], 3), np.uint8)
blank_image2[:,:] = (255,255,255)
print(hierarchy2[0][0][3])
print(hierarchy2[0][1][3])
print(hierarchy2[0][2][3])
for i in range(len(contours2)):
    if hierarchy2[0][i][3] != -1:
    # only contours htat are just under the main one
        continue
    #if cv2.arcLength(contours2[i],True) > 64:
    # draw box and diagonals
    cv2.drawContours(blank_image2, contours2, i, (0,0,0),1) 
    rect = cv2.minAreaRect(contours2[i])
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.line(blank_image2,(box[0][0],box[0][1]),(box[2][0],box[2][1]),(0,255,0),2)
    cv2.line(blank_image2,(box[1][0],box[1][1]),(box[3][0],box[3][1]),(0,255,0),2)
#    print(box[0])
    cv2.drawContours(blank_image2, [box],0, (0,255,0), 2)
    # do nothing for all the contours contained in it (=holes), and again boxes for the contours inside these, and so on
    son = hierarchy2[0][i][2]    
    while son != -1:
    # there is a contour inside -> fill it with black
        #cv2.drawContours(blank_image2, contours2, son, (0,0,0),cv2.FILLED)
        grandson = hierarchy2[0][son][2]
        while grandson != -1:
        # there is another inside -> fill with white
            cv2.drawContours(blank_image2, contours2, grandson, (0,0,0),1) 
            rect = cv2.minAreaRect(contours2[grandson])
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.line(blank_image2,(box[0][0],box[0][1]),(box[2][0],box[2][1]),(0,255,0),2)
            cv2.line(blank_image2,(box[1][0],box[1][1]),(box[3][0],box[3][1]),(0,255,0),2)
            cv2.drawContours(blank_image2, [box],0, (0,255,0), 2)
            grandson = hierarchy2[0][grandson][0] # other contours at the same level as grandson
        son = hierarchy2[0][son][0] # other contours at the same level as son
    #print("%s di %s" % (i, len(contours2)))

cv2.imwrite(contourdir+'test3.jpg', blank_image2)

