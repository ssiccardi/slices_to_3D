import numpy as np
import cv2
from optparse import OptionParser
import math
######################
# Reads an image, transforms it in gray
#  - applies the threshold with specified value
#  - applies a second adaptive threshold
#  - finds contours in the double thresholded image
#  - writes the first ncont contours (black on white) in contourdir/drawing1.jpg (ncont = input value)
#  - computes approximated poly_contours
#  - writes the first ncont poly_contours (black on white) in contourdir/drawing1b.jpg (ncont = input value)
#  - computes centroids
#  - marks centroids in green in a copy of the input image contourdir/drawing.jpg
#  - computes boxes and writes them in green in contourdir/drawing3.jpg
#  - computes Canny of the double thresholded images and saves to contourdir/drawing3b.jpg
#  - computes lines on Canny images, writes in the original image and saves to contourdir/drawing2.jpg
#  - computes lines on image with contours (drawing1.jpg), and saves to contourdir/drawing4.jpg
#  - computes lines on the once thresholded image, writes them on the original one and saves to contourdir/drawing5.jpg
#  - computes lines on the double thresholded image (no Canny), writes them on the original one and saves to contourdir/drawing6.jpg
#  - draws points inside contours delimiting edges and saves in contourdit/drawing7.jpg
#           come immagine con i puntini ma con cerchi o altri sistemi per trovare nodi o linee
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

print("Original Image "+name)
im = cv2.imread(slicedir+fname+'.png')

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

print("Computing centroids (drawing.jpg)")

# Get the moments
mu = [None]*len(contours)
for i in range(len(contours)):
    mu[i] = cv2.moments(contours[i])
# Get the mass centers
mc = [None]*len(contours)
for i in range(len(contours)):
    # add 1e-5 to avoid division by zero
    mc[i] = (mu[i]['m10'] / (mu[i]['m00'] + 1e-5), mu[i]['m01'] / (mu[i]['m00'] + 1e-5))
for i in range(len(contours)):
    color = (0, 255, 0)
    radius = cv2.pointPolygonTest(contours[i],(int(mc[i][0]),int(mc[i][1])), True)
    cv2.circle(im, (int(mc[i][0]), int(mc[i][1])), 1,color,-1)  #max(int(radius),1), color, -1)
cv2.imwrite(contourdir+'drawing.jpg', im)


# Draw contours
    
drawing = np.zeros((thresh.shape[0], thresh.shape[1], 3), dtype=np.uint8)
drawing[:,:] = (255,255,255)
    
print("Computing circles and boxes (drawing3.jpg)")

for i in range(len(contours)):
    color = (255, 0, 0)
    cv2.drawContours(drawing, contours, i, (0,0,0), 1)
    cv2.circle(drawing, (int(mc[i][0]), int(mc[i][1])), 4, color, -1)
    rect = cv2.minAreaRect(contours[i])
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(drawing, [box],0, color, 2)
cv2.imwrite(contourdir+'drawing3.jpg', drawing)



print("Computing Canny and Hough lines (drawing3b.jpg, drawing2.jpg)")

img = cv2.imread(slicedir+fname+'.png')  # new copy of input image

minLineLength = 10
maxLineGap = 1
dst = cv2.Canny(thresh, 80, 100, None, 3)
lines =  cv2.HoughLinesP(dst,1,np.pi/180,10,minLineLength,maxLineGap) # nessuna linea
for y in lines:
    x=y[0]
    cv2.line(img,(x[0],x[1]),(x[2],x[3]),(0,255,0),2)
cv2.imwrite(contourdir+'drawing2.jpg',img)
cv2.imwrite(contourdir+'drawing3b.jpg',dst)

print("Computing Hough lines on drawing1 (drawing4.jpg)")

img_blank = cv2.imread(slicedir+'drawing1.jpg')  # new copy of input image
lines1 =  cv2.HoughLinesP(img_blank,1,np.pi/180,10,minLineLength,maxLineGap) # neanche una linea
if lines1 is not None:
    for y in lines1:
        x=y[0]
        cv2.line(img_blank,(x[0],x[1]),(x[2],x[3]),(0,255,0),2)
cv2.imwrite(contourdir+'drawing4.jpg',img_blank)

#dst = cv2.Canny(thresh1, 80, 100, None, 3)

print("Computing Hough lines on single thresholded image (drawing5.jpg)")

lines2 =  cv2.HoughLinesP(thresh1,1,np.pi/180,10,minLineLength,maxLineGap) # un po' di linee con min=10, max=1
img1 = cv2.imread(slicedir+fname+'.png')  # new copy of input image
if lines2 is not None:
    for y in lines2:
        x=y[0]
        cv2.line(img1,(x[0],x[1]),(x[2],x[3]),(0,255,0),2)
cv2.imwrite(contourdir+'drawing5.jpg',img1)

print("Computing Hough lines on double thresholded image (drawing6.jpg)")

lines3 =  cv2.HoughLinesP(thresh,1,np.pi/180,10,minLineLength,maxLineGap) # tutto verde
img2 = cv2.imread(slicedir+fname+'.png')  # new copy of input image
if lines3 is not None:
    for y in lines3:
        x=y[0]
        cv2.line(img2,(x[0],x[1]),(x[2],x[3]),(0,255,0),2)
cv2.imwrite(contourdir+'drawing6.jpg',img2)

############# points inside contours ###############

print("Computing points inside contours (drawing7.jpg)")

ret, thresh2 = cv2.threshold(imgray, threshold_value, 255, cv2.THRESH_BINARY_INV)
contours2, hierarchy2 = cv2.findContours(thresh2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

blank_image2 = np.zeros((thresh2.shape[0], thresh2.shape[1], 3), np.uint8)
blank_image2[:,:] = (255,255,255)

contained_points = []
for j in range(8, 1023, 8):
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
                    if cv2.pointPolygonTest(contours2[son],(j,k), False)==1:
                        #print("son")
                        adding = False
                        break
                    else:
                        if hierarchy2[0][son][2] != -1:
                        # if the contour has a hole inside, I check that the point is not in the hole
                            grandson = hierarchy2[0][son][2]
                            #print(grandson)
                            if cv2.pointPolygonTest(contours2[grandson],(j,k), False)==1:
                                #print("grandson")
                                adding = False
                                break
                        if hierarchy2[0][son][0] != -1:
                        # if the contour has a hole inside, I check that the point is not in the hole
                            grandson = hierarchy2[0][son][0]
                            while grandson != -1:
                                #print(grandson)
                                if cv2.pointPolygonTest(contours2[grandson],(j,k), False)==1:
                                    #print("brother")
                                    adding = False
                                    break
                                grandson = hierarchy2[0][grandson][0]
            if adding == True:
                contained_points.append([j,k])
                break
# Draw contours
for i in range(len(contours2)):
    color = (0, 255, 0)
    cv2.drawContours(blank_image2, contours2, i, (0,0,0),1)
for pp in contained_points:
    cv2.circle(blank_image2,(pp[0],pp[1]), 2, color,-1)

cv2.imwrite(contourdir+'drawing7.jpg',blank_image2)
