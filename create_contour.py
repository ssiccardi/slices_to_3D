import numpy as np
import cv2
import glob
#impng = cv2.imread('Zslices/100824_drop 5 s1_z002_ch00.png')
#cv2.imwrite('Zslices/100824_drop 5 s1_z002_ch00.jpg', impng)
threshold_value = 64   # can be adjusted depending on acutal images
### threshold type = 0 -> binary. Pixel values higher than threshold are set black
### findContours approximation = NONE: all contour points are stored
slicedir = "./Zslices"
graydir = "gray"
threshdir = "thresh"
contourdir = "contour"
myslices = glob.glob(slicedir+'/*.png')
for idx, fname in enumerate(myslices):
    im = cv2.imread(fname)
    nameg = graydir+fname[len(slicedir):]
    nameg = nameg.replace('.png','.jpg')
    namet = threshdir+fname[len(slicedir):]
    namet = namet.replace('.png','.jpg')
    namec = contourdir+fname[len(slicedir):]
    namec = namec.replace('.png','.jpg')
    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
###    save grayed image in order to check how it looks like
    cv2.imwrite(nameg, imgray)
    ret, thresh = cv2.threshold(imgray, threshold_value, 255, 0)
###    save "thresholded" image in order to check how it looks like
    cv2.imwrite(namet, thresh)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    blank_image = np.zeros((1024,1024,3), np.uint8)
    blank_image[:,:] = (255,255,255)
    cv2.drawContours(blank_image, contours, -1, (0,0,0), 2)
###    save contour image to build 3D image
    cv2.imwrite(namec, blank_image)
