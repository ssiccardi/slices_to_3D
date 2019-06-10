import numpy as np
import cv2
from optparse import OptionParser
# -I image name in thresh directory
# -N number of contours to display in contour dirctory
threshold_value = 64   # can be adjusted depending on acutal images

#import glob
#impng = cv2.imread('Zslices/100824_drop 5 s1_z002_ch00.png')
#cv2.imwrite('Zslices/100824_drop 5 s1_z002_ch00.jpg', impng)
#threshold_value = 64   # can be adjusted depending on acutal images
### threshold type = 0 -> binary. Pixel values higher than threshold are set black
### findContours approximation = NONE: all contour points are stored
slicedir = "Zslices/"
graydir = "gray"
threshdir = "thresh"
contourdir = "contour/"
parser = OptionParser()
parser.add_option("-I", "--image", dest="fname",metavar="IM_NAME",
                  help="Choose Image Name")
parser.add_option("-N", "--ncontour", dest="ncont",metavar="NR_CONT",
                  help="Choose Contour Number")
(optlist, args) = parser.parse_args()
if not optlist.fname or not optlist.ncont:
	errmsg='This programs needs the image name and contour number path as a command line argument'
	raise SyntaxError(errmsg)
	

fname=optlist.fname
ncont=int(optlist.ncont)
namec=contourdir+optlist.ncont+'_'+fname+'.jpg'

print(namec)
im = cv2.imread(slicedir+fname+'.png')

imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

ret, thresh = cv2.threshold(imgray, threshold_value, 255, 0)

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
print(len(contours))

blank_image = np.zeros((1024,1024,3), np.uint8)
blank_image[:,:] = (255,255,255)
for i in range(ncont):
    cv2.drawContours(blank_image, contours, i, (0,0,0), 1, 8, hierarchy)

# Get the moments
mu = [None]*len(contours)
for i in range(len(contours)):
    mu[i] = cv2.moments(contours[i])
# Get the mass centers
mc = [None]*len(contours)
for i in range(len(contours)):
    # add 1e-5 to avoid division by zero
    mc[i] = (mu[i]['m10'] / (mu[i]['m00'] + 1e-5), mu[i]['m01'] / (mu[i]['m00'] + 1e-5))
# Draw contours
    
drawing = np.zeros((thresh.shape[0], thresh.shape[1], 3), dtype=np.uint8)
drawing[:,:] = (255,255,255)
    
for i in range(len(contours)):
    color = (0, 255, 0)
    cv2.drawContours(drawing, contours, i, (0,0,0), 1)
    cv2.circle(drawing, (int(mc[i][0]), int(mc[i][1])), 1, color, -1)

#drawing1 = np.zeros((thresh.shape[0], thresh.shape[1], 3), dtype=np.uint8)
    
#for i in range(int(len(contours)/2)):
#    color = (0, 255, 0)
#    cv2.drawContours(drawing1, contours, i, (255,255,255), 1)
#    cv2.circle(drawing1, (int(mc[i][0]), int(mc[i][1])), 1, color, -1)

###    save contour image to build 3D image
cv2.imwrite(namec, blank_image)
cv2.imwrite(contourdir+'drawing.jpg', drawing)
#cv2.imwrite(contourdir+'drawing1.jpg', drawing1)
