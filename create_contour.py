import numpy as np
import cv2
import glob
#impng = cv2.imread('Zslices/100824_drop 5 s1_z002_ch00.png')
#cv2.imwrite('Zslices/100824_drop 5 s1_z002_ch00.jpg', impng)
threshold_value = 80   # can be adjusted depending on acutal images
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
    ret, thresh = cv2.threshold(imgray, threshold_value, 255, cv2.THRESH_BINARY_INV)
###    save "thresholded" image in order to check how it looks like
    cv2.imwrite(namet, thresh)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    blank_image = np.zeros((thresh.shape[0], thresh.shape[1], 3), np.uint8)
    blank_image[:,:] = (255,255,255)
    #cv2.drawContours(blank_image, contours, -1, (0,0,0), 2)
######### moments
    # Get the moments
#    mu = [None]*len(contours)
#    for i in range(len(contours)):
#        mu[i] = cv2.moments(contours[i])
#    # Get the mass centers
#    mc = [None]*len(contours)
#    for i in range(len(contours)):
#        # add 1e-5 to avoid division by zero
#        mc[i] = (mu[i]['m10'] / (mu[i]['m00'] + 1e-5), mu[i]['m01'] / (mu[i]['m00'] + 1e-5))
############
## poly, rects and circles
    contours_poly = [None]*len(contours)
    boundRect = [None]*len(contours)
    centers = [None]*len(contours)
    radius = [None]*len(contours)
    for i, c in enumerate(contours):
        contours_poly[i] = cv2.approxPolyDP(c, 3, True)
#        boundRect[i] = cv2.boundingRect(contours_poly[i])
#        centers[i], radius[i] = cv2.minEnclosingCircle(contours_poly[i])

    # Draw contours
    for i in range(len(contours)):
        color = (0, 255, 0)
        #cv2.drawContours(blank_image, contours, i, (0,0,0), 1)
        cv2.drawContours(blank_image, contours_poly, i, (0,0,0),1)
        #cv2.rectangle(blank_image, (int(boundRect[i][0]), int(boundRect[i][1])), \
        #  (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), color, 1)
        #cv2.circle(blank_image, (int(centers[i][0]), int(centers[i][1])), int(radius[i]), color, 1)        
        #cv2.circle(blank_image, (int(mc[i][0]), int(mc[i][1])), 1, color, -1)
###########
# definisco una griglia di punti
# per ogni punto vedo se e' in un contorno con pointPolygonTest e tengo solo quelli dentro
# poi provo a collegare un punto con tutti quelli nel suo intorno e vedo se le rette intersecano i bordi
# se non li intersecano le tengo e sono i miei archi
# (questo potrei farlo anche fra 2 immagini consecutive).
# Pero' se il cammino fra due punti e' curvo 'buca' i bordi e non lo troverei.
# Oppure guardo come sono fatti i contorni
###########


###    save contour image to build 3D image
    cv2.imwrite(namec, blank_image)
