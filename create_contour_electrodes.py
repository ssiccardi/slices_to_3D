import numpy as np
import cv2
from optparse import OptionParser
import math
######################
# Reads an image, transforms it in gray
#  - adds some electrodes at fixed positions (i.e. draws withe circles)
#  - applies the threshold with specified value
#  - applies a second adaptive threshold
#  - finds contours in the double thresholded image
#  - writes the first ncont contours (black on white) in contourdir/drawing1.jpg (ncont = input value)
#
# -I image name in slices directory
# -N number of contours to display in contour dirctory
# -T threshold value (default = 80


threshold_value = 80   # can be adjusted depending on acutal images

slicedir = "Zslices/"
graydir = "gray"
threshdir = "thresh"
contourdir = "contour/"
parser = OptionParser()
parser.add_option("-I", "--image", dest="fname",metavar="IM_NAME",
                  help="Choose Image Name")
parser.add_option("-N", "--ncontour", dest="ncont",metavar="NR_CONT",
                  help="Choose Contour Number")
parser.add_option("-T", "--thvalue", dest="threshold_value",metavar="TH_VALUE",
                  help="Choose Threshold value")
(optlist, args) = parser.parse_args()
if not optlist.fname or not optlist.ncont:
	errmsg='This programs needs the image name and contour number path as a command line argument'
	raise SyntaxError(errmsg)

if not optlist.threshold_value:
	print("Using default threshold value %s ", threshold_value)
else:
	threshold_value=optlist.threshold_value

fname=optlist.fname
ncont=int(optlist.ncont)
namec=contourdir+optlist.ncont+'_'+fname+'.jpg'

print("Original Image "+fname)
im = cv2.imread(slicedir+fname+'.png')

pix = 1024 / 250 # 250 micron in 1024 pixels
radius = int(pix * 5) # 250 micron in 1024 pixels
color = (255, 255, 255)

#elect = [
#          (45*pix,60*pix),(45*pix,90*pix),(45*pix,120*pix),(45*pix,150*pix),(45*pix,180*pix),(45*pix,210*pix),
#          (75*pix,60*pix),(75*pix,90*pix),(75*pix,120*pix),(75*pix,150*pix),(75*pix,180*pix),(75*pix,210*pix),
#          (105*pix,60*pix),(105*pix,90*pix),(105*pix,120*pix),(105*pix,150*pix),(105*pix,180*pix),(105*pix,210*pix),
#          (135*pix,60*pix),(135*pix,90*pix),(135*pix,120*pix),(135*pix,150*pix),(135*pix,180*pix),(135*pix,210*pix),
#          (165*pix,60*pix),(165*pix,90*pix),(165*pix,120*pix),(165*pix,150*pix),(165*pix,180*pix),(165*pix,210*pix),
#         ]

elect = []
for k in range(5):
    for i in range(6):
        # slice 0
        if "000" in fname:
            if k==0 and i in (0,5):
                continue
            elif k==4 and i in (0, 4, 5):
                continue
        elect.append(((65+k*30)*pix,(50+i*30)*pix))
for i in range(len(elect)):
    cv2.circle(im, (int(elect[i][1]), int(elect[i][0])), radius,color,-1)  #max(int(radius),1), color, -1)

cv2.imwrite(contourdir+fname+'_elect.jpg', im)

imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

ret, thresh1 = cv2.threshold(imgray, threshold_value, 255, cv2.THRESH_TOZERO)
thresh = cv2.adaptiveThreshold(thresh1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,41, 2)

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
print("Number of contours found %s", len(contours))

blank_image = np.zeros((1024,1024,3), np.uint8)
blank_image[:,:] = (255,255,255)
blank_imageb = np.zeros((1024,1024,3), np.uint8)
blank_imageb[:,:] = (255,255,255)
contours_poly = [None]*len(contours)

print("Computing poly contours (drawing1.jpg, drawing1b.jpg)")
for i, c in enumerate(contours):
    contours_poly[i] = cv2.approxPolyDP(c, min(4,math.ceil(cv2.arcLength(c, True)*0.01)) , True)

for i in range(ncont):
    cv2.drawContours(blank_imageb, contours_poly, i, (0,0,0), 1, 8, hierarchy)
    cv2.drawContours(blank_image, contours, i, (0,0,0), 1, 8, hierarchy)

cv2.imwrite(contourdir+'drawing1.jpg', blank_image)
cv2.imwrite(contourdir+'drawing1b.jpg', blank_imageb)


