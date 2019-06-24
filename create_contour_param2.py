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
img = cv2.imread(slicedir+fname+'.png')
kernel = np.ones((5,5),np.float32)/25
src = cv2.filter2D(img,-1,kernel)

print("Applying Laplace transform (example1.jpg)")

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
cv2.imwrite(contourdir+'example1.jpg', imgResult)

# thresholding
print("Applying threshold (example2.jpg)")

bw = cv2.cvtColor(imgResult, cv2.COLOR_BGR2GRAY)
_, bw = cv2.threshold(bw, threshold_value*2, 255, cv2.THRESH_BINARY) # | cv2.THRESH_OTSU)
#bw = cv2.adaptiveThreshold(bw, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,121,2) # | cv2.THRESH_OTSU)
cv2.imwrite(contourdir+'example2.jpg', bw)


# distance transform
print("Applying distance transform on thresholded img (example3.jpg)")
dista = cv2.distanceTransform(bw, cv2.DIST_L2, 3)
# Normalize the distance image for range = {0.0, 1.0}
# so we can visualize and threshold it
dist = np.zeros(dista.shape, dtype=dista.dtype)
cv2.normalize(dista, dist, 0, 255.0, cv2.NORM_MINMAX)
cv2.imwrite(contourdir+'example3.jpg', dist)

print("Applying threshold (example4.jpg)")
# Threshold to obtain the peaks
# This will be the markers for the foreground objects
_, distb = cv2.threshold(dist, threshold_value, 255.0, cv2.THRESH_BINARY)
# Dilate a bit the dist image
kernel1 = np.ones((3,3), dtype=np.uint8)
distb = cv2.dilate(distb, kernel1)
cv2.imwrite(contourdir+'example4.jpg', distb)

print("Finding markers (example5.jpg)")
# Create the CV_8U version of the distance image
# It is needed for findContours()
dist_8u = distb.astype('uint8')
# Find total markers
contours, _ = cv2.findContours(dist_8u, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# Create the marker image for the watershed algorithm
markers = np.zeros(distb.shape, dtype=np.int32)
# Draw the foreground markers
for i in range(len(contours)):
    cv2.drawContours(markers, contours, i, (i+1), -1)
# Draw the background marker
cv2.circle(markers, (5,5), 3, (255,255,255), -1)
cv2.imwrite(contourdir+'example5.jpg', markers*10000)

print("Drawing circles in markers (example5b.jpg)")
print(contours[0])
for i in range(len(contours)):
    mask = np.zeros((markers.shape[0], markers.shape[1]), dtype=np.ubyte)
    cv2.drawContours(mask,contours,i,1,cv2.FILLED)
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(dista, mask)
    #cv2.drawContours(src, contours, i, (255,255,255), -1)
    cv2.circle(src, maxLoc, int(maxVal)+1,(0,255,0),1)
cv2.imwrite(contourdir+'example5b.jpg', src)


print("Performing watershed (example6.jpg, example7.jpg)")
# Perform the watershed algorithm
cv2.watershed(imgResult, markers)
#mark = np.zeros(markers.shape, dtype=np.uint8)
mark = markers.astype('uint8')
mark = cv2.bitwise_not(mark)
# uncomment this if you want to see how the mark
# image looks like at that point
cv2.imwrite(contourdir+'example6.jpg', mark)
# Generate random colors
colors = []
for contour in contours:
    colors.append((rng.randint(0,256), rng.randint(0,256), rng.randint(0,256)))
# Create the result image
dst = np.zeros((markers.shape[0], markers.shape[1], 3), dtype=np.uint8)
# Fill labeled objects with random colors
for i in range(markers.shape[0]):
    for j in range(markers.shape[1]):
        index = markers[i,j]
        if index > 0 and index <= len(contours):
            dst[i,j,:] = colors[index-1]
# Visualize the final image
cv2.imwrite(contourdir+'example7.jpg', dst)

##################
#
#  New elaboration, restarting from the contours
#
# distance transform directly on the thresholded img does not find anything
# I try on the contours

print("Finding contours (example8.jpg, example8b.jpg)")
img = cv2.imread(slicedir+fname+'.png')
kernel = np.ones((5,5),np.float32)/25
src = cv2.filter2D(img,-1,kernel)

imgray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

ret, thresh2 = cv2.threshold(imgray, threshold_value, 255, cv2.THRESH_BINARY_INV)
contours2, hierarchy2 = cv2.findContours(thresh2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
print(len(contours2))

blank_image2 = np.zeros((src.shape[0], src.shape[1], 3), np.uint8)
#blank_image2[:,:] = (255,255,255)

for i in range(1,len(contours2)):
    if hierarchy2[0][i][3] != 0:
    # only contours htat are just under the main one
        continue
    #if cv2.arcLength(contours2[i],True) > 64:
    # fill the contour with white
    cv2.drawContours(blank_image2, contours2, i, (255,255,255),cv2.FILLED) 
    # fill with black all the contours contained in it (=holes), and again with white the contours inside these, and so on
    son = hierarchy2[0][i][2]    
    while son != -1:
    # there is a contour inside -> fill it with black
        cv2.drawContours(blank_image2, contours2, son, (0,0,0),cv2.FILLED)
        grandson = hierarchy2[0][son][2]
        while grandson != -1:
        # there is another inside -> fill with white
            cv2.drawContours(blank_image2, contours2, grandson, (255,255,255),cv2.FILLED) 
            grandson = hierarchy2[0][grandson][0] # other contours at the same level as grandson
        son = hierarchy2[0][son][0] # other contours at the same level as son
    #print("%s di %s" % (i, len(contours2)))

cv2.imwrite(contourdir+'example8.jpg', blank_image2)

print("Computing Hough lines on contoured image (drawing8c.jpg)")
blank_image2_u = cv2.cvtColor(blank_image2, cv2.COLOR_BGR2GRAY)
lines2 =  cv2.HoughLinesP(blank_image2_u,1,np.pi/180,10,10,1) # un po' di linee con min=10, max=1
img1 = cv2.imread(slicedir+fname+'.png')  # new copy of input image
if lines2 is not None:
    for y in lines2:
        x=y[0]
        cv2.line(img1,(x[0],x[1]),(x[2],x[3]),(0,255,0),2)
cv2.imwrite(contourdir+'example8c.jpg',img1)

# distance transform
bw1 = cv2.cvtColor(blank_image2, cv2.COLOR_BGR2GRAY)
_, bw1 = cv2.threshold(bw1, threshold_value, 255, cv2.THRESH_BINARY)# | cv2.THRESH_OTSU)



cv2.imwrite(contourdir+'example8b.jpg', bw1)

#bw = np.uint8(blank_image2)

print("Applying distance transform on contour img (example9.jpg)")
dist1a = cv2.distanceTransform(bw1, cv2.DIST_L2, 3)
# Normalize the distance image for range = {0.0, 1.0}
# so we can visualize and threshold it
dist1 = np.zeros(dist1a.shape, dtype=dist1a.dtype)
cv2.normalize(dist1a, dist1, 0, 255.0, cv2.NORM_MINMAX)
cv2.imwrite(contourdir+'example9.jpg', dist1)


# Threshold to obtain the peaks
print("Applying threshold to dist transformed (example10.jpg)")
# This will be the markers for the foreground objects
_, dist1 = cv2.threshold(dist1, threshold_value, 255, cv2.THRESH_BINARY)
# Dilate a bit the dist image
kernel1 = np.ones((3,3), dtype=np.uint8)
dist1 = cv2.dilate(dist1, kernel1)
cv2.imwrite(contourdir+'example10.jpg', dist1)

print("Creating markers (example11.jpg)")

# Create the CV_8U version of the distance image
# It is needed for findContours()
dist_8u1 = dist1.astype('uint8')
# Find total markers
contours1, _ = cv2.findContours(dist_8u1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# Create the marker image for the watershed algorithm
markers1 = np.zeros(dist1.shape, dtype=np.int32)
# Draw the foreground markers
for i in range(len(contours1)):
    cv2.drawContours(markers1, contours1, i, (i+1), -1)
# Draw the background marker
cv2.circle(markers1, (5,5), 3, (255,255,255), -1)
cv2.imwrite(contourdir+'example11.jpg', markers1)


print("Drawing circles in markers (example11b.jpg, example11c.jpg)")
print(contours1[0])
src8c = cv2.imread(contourdir+'example8.jpg')

nodes = []
for i in range(len(contours1)):
    mask = np.zeros((markers1.shape[0], markers1.shape[1]), dtype=np.ubyte)
    cv2.drawContours(mask,contours1,i,1,cv2.FILLED)
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(dist1a, mask)
    #cv2.drawContours(src, contours1, i, (255,255,255), -1)
    cv2.circle(src, maxLoc, int(maxVal)+1,(0,255,0),1)
    cv2.circle(src8c, maxLoc, int(maxVal)+1,(255,0,0),1)
    nodes.append(maxLoc)
for i in range(len(nodes)):
    for k in range(i+1,len(nodes)):
    ## how to find a path between two nodes, without going out of the white mask?
        midi = (int((nodes[i][0]+nodes[k][0])*0.5),int((nodes[i][1]+nodes[k][1])*0.5))
        midi2 = (int((nodes[i][0]+midi[0])*0.5),int((nodes[i][1]+midi[1])*0.5))
        midi3 = (int((midi[0]+nodes[k][0])*0.5),int((midi[1]+nodes[k][1])*0.5))
        #cv2.circle(src8c, midi, 2,(255,0,0),1)
        #cv2.circle(src8c, midi2, 2,(255,0,0),1)
        #cv2.circle(src8c, midi3, 2,(255,0,0),1)
        a11 = src8c[int(midi[0]),int(midi[1]),1]
        a21 = src8c[int(midi2[0]),int(midi2[1]),1]
        a31 = src8c[int(midi3[0]),int(midi3[1]),1]
        a10 = src8c[int(midi[0]),int(midi[1]),0]
        a20 = src8c[int(midi2[0]),int(midi2[1]),0]
        a30 = src8c[int(midi3[0]),int(midi3[1]),0]
        a12 = src8c[int(midi[0]),int(midi[1]),2]
        a22 = src8c[int(midi2[0]),int(midi2[1]),2]
        a32 = src8c[int(midi3[0]),int(midi3[1]),2]
        if (a11 == 255) and (a21 == 255) and (a31 == 255) and (a10 == 255) and (a20 == 255) and (a30 == 255) and (a12 == 255) and (a22 == 255) and (a32 == 255):
            cv2.circle(src8c, midi, 2,(255,0,0),1)
            cv2.circle(src8c, midi2, 2,(255,0,0),1)
            cv2.circle(src8c, midi3, 2,(255,0,0),1)
            cv2.line(src8c,nodes[i],nodes[k],(255,0,0),1)
cv2.imwrite(contourdir+'example11b.jpg', src)
cv2.imwrite(contourdir+'example11c.jpg', src8c)

print("Performing watershed (example12.jpg, example13.jpg)")

# Perform the watershed algorithm
cv2.watershed(imgResult, markers1)
#cv2.watershed(imgray, markers1)
#mark = np.zeros(markers.shape, dtype=np.uint8)
mark1 = markers1.astype('uint8')
mark1 = cv2.bitwise_not(mark1)
# uncomment this if you want to see how the mark
# image looks like at that point
cv2.imwrite(contourdir+'example12.jpg', mark1)
# Generate random colors
colors = []
for contour in contours1:
    colors.append((rng.randint(0,256), rng.randint(0,256), rng.randint(0,256)))
# Create the result image
dst1 = np.zeros((markers1.shape[0], markers1.shape[1], 3), dtype=np.uint8)
# Fill labeled objects with random colors
for i in range(markers1.shape[0]):
    for j in range(markers1.shape[1]):
        index = markers1[i,j]
        if index > 0 and index <= len(contours1):
            dst1[i,j,:] = colors[index-1]
# Visualize the final image
cv2.imwrite(contourdir+'example13.jpg', dst1)
