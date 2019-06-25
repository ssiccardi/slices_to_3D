import numpy as np
import cv2
from optparse import OptionParser
import math
import random as rng
######################
# 
#
# - Blurs image, saved as contourdir/test1.jpg
# - applies threshold, finds contours and draws diagonals of rectangles, saved as contourdir/test2.jpg
# - computes approximated (poly) contours, saved as contourdir/test2b.jpg
# - finds points in poly contours (one each 8 pixels), saved as contourdir/test3.jpg, test3b.jpg (shrinked)
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

print("Blurring (test1.jpg)")

kernel = np.ones((5,5),np.float32)/25
src = cv2.filter2D(img,-1,kernel)

cv2.imwrite(contourdir+'test1.jpg', src)

#print("Applying Laplace transform (test.jpg)")

#kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float32)
# do the laplacian filtering as it is
# well, we need to convert everything in something more deeper then CV_8U
# because the kernel has some negative values,
# and we can expect in general to have a Laplacian image with negative values
# BUT a 8bits unsigned int (the one we are working with) can contain values from 0 to 255
# so the possible negative number will be truncated
#imgLaplacian = cv2.filter2D(src, cv2.CV_32F, kernel)
#sharp = np.float32(src)
#imgResult = sharp - imgLaplacian
# convert back to 8bits gray scale
#imgResult = np.clip(imgResult, 0, 255)
#imgResult = imgResult.astype('uint8')
#imgLaplacian = np.clip(imgLaplacian, 0, 255)
#imgLaplacian = np.uint8(imgLaplacian)
#cv.imshow('Laplace Filtered Image', imgLaplacian)
#cv2.imshow('New Sharped Image', imgResult)
#cv2.imwrite(contourdir+'test1.jpg', imgResult)

# thresholding

####### non uso la trasformata...

imgray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

ret, thresh2 = cv2.threshold(imgray, threshold_value, 255, cv2.THRESH_BINARY_INV)
contours2, hierarchy2 = cv2.findContours(thresh2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

blank_image2 = np.zeros((thresh2.shape[0], thresh2.shape[1], 3), np.uint8)
blank_image2[:,:] = (255,255,255)



print("Applying threshold, finding contours and writing main boxes (test2.jpg)")

print("%s contours found" % len(contours2))

for i in range(1,len(contours2)):
    if hierarchy2[0][i][3] != 0:
    # only contours that are just under the main one
        continue
    # draw box and diagonals
    cv2.drawContours(blank_image2, contours2, i, (0,0,0),cv2.FILLED) 
    if cv2.arcLength(contours2[i],True)>32:
        rect = cv2.minAreaRect(contours2[i])
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.line(blank_image2,(box[0][0],box[0][1]),(box[2][0],box[2][1]),(0,255,0),1)
        cv2.line(blank_image2,(box[1][0],box[1][1]),(box[3][0],box[3][1]),(0,255,0),1)
#       print(box[0])
        cv2.drawContours(blank_image2, [box],0, (0,255,0), 1)
    # do nothing for all the contours contained in it (=holes), and again boxes for the contours inside these, and so on
    son = hierarchy2[0][i][2]    
    while son != -1:
    # there is a contour inside -> fill it with black
        cv2.drawContours(blank_image2, contours2, son, (256,256,256),cv2.FILLED)
        grandson = hierarchy2[0][son][2]
        while grandson != -1:
        # there is another inside -> fill with white
            cv2.drawContours(blank_image2, contours2, grandson, (0,0,0),cv2.FILLED) 
            #rect = cv2.minAreaRect(contours2[grandson])
            #box = cv2.boxPoints(rect)
            #box = np.int0(box)
            #cv2.line(blank_image2,(box[0][0],box[0][1]),(box[2][0],box[2][1]),(0,255,0),2)
            #cv2.line(blank_image2,(box[1][0],box[1][1]),(box[3][0],box[3][1]),(0,255,0),2)
            #cv2.drawContours(blank_image2, [box],0, (0,255,0), 2)
            grandson = hierarchy2[0][grandson][0] # other contours at the same level as grandson
        son = hierarchy2[0][son][0] # other contours at the same level as son
    #print("%s di %s" % (i, len(contours2)))

cv2.imwrite(contourdir+'test2.jpg', blank_image2)

blank_imageb = np.zeros((1024,1024,3), np.uint8)
blank_imageb[:,:] = (255,255,255)
contours_poly = [None]*len(contours2)

print("Computing poly contours (test2b.jpg)")
for i, c in enumerate(contours2):
    contours_poly[i] = cv2.approxPolyDP(c, min(4,math.ceil(cv2.arcLength(c, True)*0.01)) , True)

for i in range(1,len(contours2)):
    if hierarchy2[0][i][3] != 0:
    # only contours that are just under the main one
        continue
    # draw box and diagonals
#    if cv2.arcLength(contours2[i],True)>32:
    cv2.drawContours(blank_imageb, contours_poly, i, (0,0,0),cv2.FILLED) 
    # do nothing for all the contours contained in it (=holes), and again boxes for the contours inside these, and so on
    son = hierarchy2[0][i][2]    
    while son != -1:
    # there is a contour inside -> fill it with black
        cv2.drawContours(blank_imageb, contours_poly, son, (256,256,256),cv2.FILLED)
        grandson = hierarchy2[0][son][2]
        while grandson != -1:
        # there is another inside -> fill with white
            cv2.drawContours(blank_imageb, contours_poly, grandson, (0,0,0),cv2.FILLED) 
            grandson = hierarchy2[0][grandson][0] # other contours at the same level as grandson
        son = hierarchy2[0][son][0] # other contours at the same level as son


cv2.imwrite(contourdir+'test2b.jpg', blank_imageb)


blank_image2 = np.zeros((thresh2.shape[0], thresh2.shape[1], 3), np.uint8)
blank_image2[:,:] = (255,255,255)

short_image = np.zeros((128, 128, 3), np.uint8)
#short_image[:,:] = (255,255,255)

print("Finding contained points (test3.jpg, test3b.jpg)")
jj = 0
kk = 0
contained_points = []
for j in range(8, 1023, 8):
    kk = 0
    for k in range(8, 1023, 8):
        adding=False
        for i in range(1,len(contours2)):
            if hierarchy2[0][i][3] != 0:
            # only contours htat are just under the main one
                continue
            if cv2.pointPolygonTest(contours2[i],(j,k), False)==1:
                    #print(hierarchy[0][i],i)
                adding = True
                if hierarchy2[0][i][2] != -1:
                # if the contour has a hole inside, I check that the point is not in the hole
                    son = hierarchy2[0][i][2]
                    #print(hierarchy[0][son],son,"s")
                    while son != -1:
                        if cv2.pointPolygonTest(contours2[son],(j,k), False)==1:
                            #print("son")
                            adding = False
                            break
                        else:
                            # if the contour has a hole inside, I check that the point is not in the hole
                            grandson = hierarchy2[0][son][2]
                            #print(grandson)
                            while grandson != -1:
                                if cv2.pointPolygonTest(contours2[grandson],(j,k), False)==1:
                                    #print("grandson")
                                    adding = False
                                    break
                                grandson = hierarchy2[0][grandson][0]
                        son = hierarchy2[0][son][0]
                if adding == False:
                    break
            if adding == True:
                contained_points.append([j,k,i])
                short_image[kk][jj] = (255,255,255)
                break
        kk = kk + 1
    jj = jj + 1
# Draw contours
#for i in range(len(contours2)):
#    color = (0, 255, 0)
#    cv2.drawContours(blank_image2, contours2, i, (0,0,0),1)
i = contained_points[0][2]
color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
for pp in contained_points:
    if i != pp[2]:
        i = pp[2]
        color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
    cv2.circle(blank_image2,(pp[0],pp[1]), 2, color,-1)
#    short_image[int(pp[0]/8), int(pp[1]/8)] = (0,0,0)
cv2.imwrite(contourdir+'test3.jpg', blank_image2)
#cv2.imwrite(contourdir+'test3b.jpg', cv2.cvtColor(short_image, cv2.COLOR_BGR2GRAY))
cv2.imwrite(contourdir+'test3b.jpg', short_image)

print("Computing Hough lines on contoured image (test4.jpg)")
short_image2_u = cv2.cvtColor(short_image, cv2.COLOR_BGR2GRAY)
lines2 =  cv2.HoughLinesP(short_image2_u,1,np.pi/60,4,4,0) # un po' di linee con min=5, max=1
short_image2 = np.zeros((128, 128, 3), np.uint8)
short_image2[:,:] = (255,255,255)

if lines2 is not None:
    print("Found %s Lines" % len(lines2))
    for y in lines2:
        x=y[0]
        cv2.line(short_image2,(x[0],x[1]),(x[2],x[3]),(0,0,0),1)
cv2.imwrite(contourdir+'test4.jpg',short_image2)

