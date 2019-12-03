import numpy as np
import cv2
from optparse import OptionParser
import math
import random as rng
import io
import glob

import xlwt

######################
# Reads all images in input directory and performs the following:
#
# Elaboration:
# - reads input file and extracts z value = nn looking for _z0nn_
# - finds contours
# - applies distance transform, saved as gray/nn.jpg
# - applies threshold to dst transformed, saved as thresh/nn.jpg
# - finds markers and lines (straight lines + ellipses) joining them, saved as contourdir/nn.jpg,orig_nn.jpg
# - merges 2 consecutive contours, and erodes them (to mimik bundles along the z axes), finds straight lines joining nodes
#
# Creates structures:
# - nodes: [index, z, (x,y), width, [linked to], [linked - triangles], subgraph]
# - lines: [(x,y) of P1, (x,y) of P2, length, index of P1, index of P2, width, [intermediate points], type, paramenters forr ellipses, status, subgraph]
# Width of a node = radius of circle centerd at the node and contained in the contour
# Widh of a line = average width of the nodes it joins
# Parameters
# -I image name in slices directory
# -T threshold value (default = 80, use 70 for slide 0)



threshold_value = 80   # 70 for slide 0 # can be adjusted depending on acutal images
pix = 1024 / 250 # 250 micron in 1024 pixels
deptz = int(pix*110 / 29) # hypothetical number of pixels between 2 images: 29 images in 110 micron

slicedir = "Zslices/"
graydir = "gray/"
threshdir = "thresh/"
contourdir = "contour/"
datadir = "data/"
parser = OptionParser()
parser.add_option("-T", "--thvalue", dest="threshold_value",metavar="TH_VALUE",
                  help="Choose Threshold value")
parser.add_option("-N", "--nimage", dest="im_elect",
                  help="Choose Image Number holding electrodes")
(optlist, args) = parser.parse_args()

if not optlist.im_elect:
	errmsg='This programs needs the number of the image holding electrodes as a command line argument'
	raise SyntaxError(errmsg)
else:
    im_elect = int(optlist.im_elect)


if not optlist.threshold_value:
    print("Using default threshold value %s " % threshold_value)
else:
    threshold_value=float(optlist.threshold_value)

myslices = sorted(glob.glob(slicedir+'/*.png'))

indnode = 1  # Node index and structures, common to all images
allnodes = []
alllines = []
xlsname = "network_"+str(im_elect)+".xls"

radius = int(pix * 5) # 250 micron in 1024 pixels
color = (255, 255, 255)
elect = []
for k in range(5):
    for i in range(6):
        # slice 0: it is useless to skip positions: if they do not touch any bundle they are not considered
        #if im_elect == 0:
        #    if k==0 and i in (0,5):
        #        continue
        #    elif k==4 and i in (0, 4, 5):
        #        continue
        # x and y, electrode name, place to store node's id
        elect.append([(65+k*30)*pix,(50+i*30)*pix,str(k+1)+str(i+1),-1])
        # name: 1st digit = column (1-5) + row (1-6)


prev_img = None
prev_nodes = None
first_img = True

for idx, fname in enumerate(myslices):

    print("Original Image "+fname)
    if not "_z0" in fname:
        errmsg = "Image name must contain z value in the form _z0nn_"
        raise SyntaxError(errmsg)

    xxx = fname.find("_z0")
    zname = fname[xxx+3:xxx+5]+"_el"+str(im_elect)+".jpg"
    zval = int(fname[xxx+3:xxx+5])
    print("Output saved as "+zname)


    print("Finding contours")
    img = cv2.imread(fname)
    kernel = np.ones((5,5),np.float32)/25
    src = cv2.filter2D(img,-1,kernel)

    imgray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    ret, thresh2 = cv2.threshold(imgray, threshold_value, 255, cv2.THRESH_BINARY_INV)
#    if zval == im_elect:
#        for i in range(len(elect)):
#            cv2.circle(thresh2, (int(elect[i][1]), int(elect[i][0])), radius,(255, 255, 255),-1)  #max(int(radius),1), color, -1)
    contours2, hierarchy2 = cv2.findContours(thresh2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    blank_image2 = np.zeros((src.shape[0], src.shape[1], 3), np.uint8)

    for i in range(1,len(contours2)):
        if hierarchy2[0][i][3] != 0:
        # only contours htat are just under the main one
            continue
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



    # distance transform
    bw1 = cv2.cvtColor(blank_image2, cv2.COLOR_BGR2GRAY)
    _, bw1 = cv2.threshold(bw1, threshold_value, 255, cv2.THRESH_BINARY)# | cv2.THRESH_OTSU)

    if zval == im_elect:
        for i in range(len(elect)):
            cv2.circle(bw1, (int(elect[i][1]), int(elect[i][0])), radius,(255, 255, 255),-1)  #max(int(radius),1), color, -1)


    #cv2.imwrite(contourdir+'example8b.jpg', bw1)

    #bw = np.uint8(blank_image2)

    print("Applying distance transform on contour img (graydir/nn.jpg)")
    dist1a = cv2.distanceTransform(bw1, cv2.DIST_L2, 3)
    # Normalize the distance image for range = {0.0, 1.0}
    # so we can visualize and threshold it
    dist1 = np.zeros(dist1a.shape, dtype=dist1a.dtype)
    cv2.normalize(dist1a, dist1, 0, 255.0, cv2.NORM_MINMAX)
    cv2.imwrite(graydir+zname, dist1)


    # Threshold to obtain the peaks
    print("Applying threshold to dist transformed (threshold/nn.jpg)")
    # This will be the markers for the foreground objects
    _, dist1 = cv2.threshold(dist1, int(threshold_value*0.9), 255, cv2.THRESH_BINARY)
    # Dilate a bit the dist image
    kernel1 = np.ones((3,3), dtype=np.uint8)
    dist1 = cv2.dilate(dist1, kernel1)
    cv2.imwrite(threshdir+zname, dist1)

    print("Creating markers")

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

    print("Drawing circles in markers (contour/nn.jpg, contour/orig_nn.jpg)")
    src8c = blank_image2
    src8out = np.copy(blank_image2)
    src8out[src8out<10] = 102  # gray backgroud for easier reading

    nodes = []
    # - nodes: [index, z, (x,y), width, [linked to],[linked - triangles]]
    #indnode = 1 global index, set at the beginning!
    firstnode = indnode  # first node number of this image
    # store nodes
    for i in range(len(contours1)):
        mask = np.zeros((markers1.shape[0], markers1.shape[1]), dtype=np.ubyte)
        cv2.drawContours(mask,contours1,i,1,cv2.FILLED)
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(dist1a, mask)
        if zval == im_elect:
        # check if the node is an electrode and mark it
            for ele in elect:
                if math.sqrt(math.pow(maxLoc[0]-int(ele[1]),2)+math.pow(maxLoc[1]-int(ele[0]),2))<= radius:
                    typnode = "E"
                    namnode = ele[2]
                    ele[3] = indnode
                    break
                else:
                    typnode = "N"
                    namnode = ""
        else:
            typnode = "N"
            namnode = ""
        nodes.append([indnode,zval, maxLoc, int(maxVal), [], [],0,typnode,namnode])
        indnode = indnode + 1
    # ensure that all the electrodes have an associated node
    if zval == im_elect:
        for ele in elect:
            if ele[3] == -1:
            # assign a node to this electrode that has been skipped
                nodes.append([indnode,zval, (int(ele[1]),int(ele[0])), int(radius), [], [],0,"E",ele[2]])
                ele[3] =indnode
                indnode = indnode + 1
    totlines = []
    print("Found %s nodes" % len(nodes))
    for i in range(len(nodes)):
        for k in range(i+1,len(nodes)):
            toler = 4
        ## how to find a path between two nodes, without going out of the white mask?
        ## ADD some other middel points to avoid bridging two contours
        ## TRY also elliptial curves if no straight lines can do
            dd = math.sqrt(math.pow(nodes[i][2][0]-nodes[k][2][0],2)+math.pow(nodes[i][2][1]-nodes[k][2][1],2))
            minax = dd/18
            nx = math.floor(dd/16)
            lineok = True
            circleok = False
            midi_nodes = []
            center = (int((nodes[i][2][0] + nodes[k][2][0])/2), int((nodes[i][2][1] + nodes[k][2][1])/2))
            if nodes[i][2][0] == nodes[k][2][0]:
                alpha = np.pi/2
            else:
                alpha = math.atan2(nodes[i][2][1]-nodes[k][2][1],nodes[i][2][0]-nodes[k][2][0])
            versus = 1
            if nx > 0:
                if nodes[i][2][0] > nodes[k][2][0]:
                    inx = nodes[k][2][0]
                    iny = nodes[k][2][1]
                    dx = (nodes[i][2][0] - nodes[k][2][0])/nx
                    dy = (nodes[i][2][1] - nodes[k][2][1])/nx
                else:
                    inx = nodes[i][2][0]
                    iny = nodes[i][2][1]
                    dx = (nodes[k][2][0] - nodes[i][2][0])/nx
                    dy = (nodes[k][2][1] - nodes[i][2][1])/nx
                for iix in range(nx):
                    midi = (int(inx+dx*(iix+1)), int(iny+dy*(iix+1)))
                    midi_nodes.append(midi)
                    # test that points on the line are in the contour, with tolerance
                    found = False
                    for iir in range(-toler, toler+1):
                        for iis in range(-toler, toler+1):
                            a10 = src8c[midi[1]+iir,midi[0]+iis,0]
                            a11 = src8c[midi[1]+iir,midi[0]+iis,1]
                            a12 = src8c[midi[1]+iir,midi[0]+iis,2]
                            if (a10 == 255) and (a11 == 255) and (a12 == 255):
                            # at least one point in the neighbour belongs to the contour
                                found = True
                                break
                        if found == True:
                            break
                    if found == False:
                    # NO points of the neighbor belong to the contour: discard the line
                        lineok = False
                        break
                # look for elliptical lines if the points are not too far from each other
                if (lineok == False) and (dd <200):
                    toler = 1
                    for iid in range(1,10):
                        minax = dd/18*iid
                        lineok = True
                        circleok = True
                        midi_nodes = []
                        tmp_nodes = cv2.ellipse2Poly(center,(int(dd/2),int(minax)),int(alpha*180/np.pi),0,180,20)
                        for mmm in tmp_nodes:
                            midi = (mmm[0], mmm[1])
                            midi_nodes.append(midi)
                            found = False
                            for iir in range(-toler, toler+1):
                                for iis in range(-toler, toler+1):
                                    a10 = src8c[midi[1]+iir,midi[0]+iis,0]
                                    a11 = src8c[midi[1]+iir,midi[0]+iis,1]
                                    a12 = src8c[midi[1]+iir,midi[0]+iis,2]
                                    if (a10 == 255) and (a11 == 255) and (a12 == 255):
                                    # at least one point in the neighbour belongs to the contour
                                        found = True
                                        break
                                if found == True:
                                    break
                            if found == False:
                            # NO points of the neighbor belong to the contour: discard the line
                                lineok = False
                                circleok = False
                                break
                        if lineok == True:
                            versus = 1
                            break
                        # the other half of the ellipse
                        lineok = True
                        circleok = True
                        midi_nodes = []
                        tmp_nodes = cv2.ellipse2Poly(center,(int(dd/2),int(minax)),int(alpha*180/np.pi),-180,0,20)
                        for mmm in tmp_nodes:
                            midi = (mmm[0], mmm[1])
                            midi_nodes.append(midi)
                            found = False
                            for iir in range(-toler, toler+1):
                                for iis in range(-toler, toler+1):
                                    a10 = src8c[midi[1]+iir,midi[0]+iis,0]
                                    a11 = src8c[midi[1]+iir,midi[0]+iis,1]
                                    a12 = src8c[midi[1]+iir,midi[0]+iis,2]
                                    if (a10 == 255) and (a11 == 255) and (a12 == 255):
                                    # at least one point in the neighbour belongs to the contour
                                        found = True
                                        break
                                if found == True:
                                    break
                            if found == False:
                            # NO points of the neighbor belong to the contour: discard the line
                                lineok = False
                                circleok = False
                                break
                        if lineok == True:
                            versus = -1
                            break

            if lineok == True:
    # - lines: [(x,y) of P1, (x,y) of P2, length, index of P1, index of P2, width, [intermediate points], type, paramenters forr ellipses, status]
    #          width is estimated as average width of its endpoints if neither is an electrode; the non-electrode end width otherwise
                l_corr = 0   # we subtract the electrode's radius from the line length, but if the resulting length is <=0 we adjust it somehow
                if nodes[k][7] == 'E':
                    wline = int(nodes[i][3])
                    l_corr = l_corr + radius
                elif nodes[i][7] == 'E':
                    wline = int(nodes[k][3])
                    l_corr = l_corr + radius
                else:
                    wline = int((nodes[k][3]+nodes[i][3])/2)
                if nodes[k][7] == 'E' and nodes[i][7] == 'E':
                    l_corr = radius * 2
                if circleok == False:
                    type = 'line'
                    leng = dd - l_corr # distance of the 2 points
                    if leng <=0:
                        leng = dd - l_corr
                    if leng <=0:
                        leng = dd
                else:
                    type = 'ellipse'
                    ltemp = np.pi * ( 3*(dd/2+minax) - np.sqrt( (3*dd/2 + minax) * (dd/2 + 3*minax) ) ) / 2 # approxinate half perimeter
                    leng = ltemp - l_corr
                    if leng <=0:
                        leng = ltemp - l_corr
                    if leng <=0:
                        leng = ltemp
                
                nodes[i][4].append(nodes[k][0])
                nodes[k][4].append(nodes[i][0])
                totlines.append({'p1': nodes[i][2], 'p2': nodes[k][2], 'len': leng, 'width': wline, 'ip1glob': nodes[i][0], 'ip2glob': nodes[k][0], 'ip1': i, 'ip2':k, 'midi': midi_nodes, 'type': type, 'center': center, 'maxax': dd/2, 'minax': minax, 'alpha':alpha, 'versus': versus, 'status': 'ok', 'subgraph': 0})
    print("Found %s lines" % len(totlines))

    # draw nodes
    for node in nodes:
        cv2.circle(src, node[2], node[3]+1,(255,255,255),1)
        cv2.circle(src8out, node[2], node[3]+1,(255,0,0),1)
#        cv2.putText(src8out,str(node[0]),node[2], cv2.FONT_HERSHEY_PLAIN, 1,(0,0,0),1,cv2.LINE_AA) moved later for readability
    # delete from totlines the third edge of triangles if any, setting status=triang
        if node[7] == "E":
            continue  # always leave there the electrodes
        for idson in node[4]:
            if idson < node[0]:
            # do not consider nodes before the one at hand
                continue
            son = next(nnn for nnn in nodes if nnn[0] == idson)
            if son[7] == "E":
                continue  # always leave there the electrodes
            for idgson in son[4]:
                if idgson < son[0]:
                    continue              
                gson = next(nnn for nnn in nodes if nnn[0] == idgson)
                if gson[7] == "E":
                    continue  # always leave there the electrodes
                if idgson in node[4]:
                    todelete = next(lll for lll in totlines if (lll['ip1glob'] ==node[0] and lll['ip2glob']==idgson) or (lll['ip2glob'] ==node[0] and lll['ip1glob']==idgson))
                    todelete['status'] = 'triang'

    # draw edges, markers, etc. Write nodes links without triangles
    last_node = -1
    link_no_tr = []
    for line in totlines:
        if line['status'] == 'triang':
            continue
        if line['type'] == 'line':
            cv2.line(src8out,line['p1'],line['p2'],(0,0,255),2)
            cv2.line(src,line['p1'],line['p2'],(255,255,255),2)
        else:
            if line['versus'] ==1:
                cv2.ellipse(src8out,line['center'],(int(line['maxax']),int(line['minax'])),line['alpha']*180/np.pi,0,180,(0,0,255),2)
                cv2.ellipse(src,line['center'],(int(line['maxax']),int(line['minax'])),line['alpha']*180/np.pi,0,180,(255,255,255),2)
            else:
                cv2.ellipse(src8out,line['center'],(int(line['maxax']),int(line['minax'])),line['alpha']*180/np.pi,-180,0,(0,0,255),2)
                cv2.ellipse(src,line['center'],(int(line['maxax']),int(line['minax'])),line['alpha']*180/np.pi,-180,0,(255,255,255),2)
        for m in line['midi']:
            cv2.circle(src8out, m, 2,(0,255,0),-1)
            cv2.circle(src, m, 2,(255,255,255),-1)
        nodes[line['ip2']][5].append(line['ip1glob'])
        if line['ip1'] != last_node:
            if last_node != -1:
                if link_no_tr:
                    nodes[last_node][5].extend(link_no_tr)
            last_node = line['ip1']
            link_no_tr = []
        link_no_tr.append(line['ip2glob'])
    if last_node:
        if link_no_tr:
            nodes[last_node][5].extend(link_no_tr)

    for node in nodes:
        cv2.putText(src8out,str(node[0]),node[2], cv2.FONT_HERSHEY_PLAIN, 1,(0,0,0),1,cv2.LINE_AA)

    cv2.imwrite(contourdir+'orig_'+zname, src)
    cv2.imwrite(contourdir+zname, src8out)

    alllines.extend(totlines)  # add edges found so far

    if not first_img:
        print("computing edges between img %s and %s" % (zval, zval-1))
        merged_image = np.zeros((src.shape[0], src.shape[1], 3), np.uint8)
        merged_image[:,:] = (102,102,102)
        kkk = np.ones((7,7),np.uint8)
        mask = cv2.erode(prev_img, kkk, iterations=1) > 200
        merged_image[mask] = 255
        mask = cv2.erode(src8c, kkk, iterations=1) > 200
        merged_image[mask] = 255
        mergedout = np.copy(merged_image)

        # 2 conotour images merged, compute edges between all nodes of the 1st and all node of the 2nd that stay within contours. Tolerance is reduced

        totlines = []
        for i in range(len(nodes)):
        # if the electrodes are not in the 1st slide, we connect nodes of the slide before the one with electrodes only to electrodes
        # for slides far from the electrodes we manage all potential connections as usual
            if zval == im_elect:
                if nodes[i][7] != "E":
                    continue
            for k in range(len(prev_nodes)):
                toler = 1
                dd = math.sqrt(math.pow(nodes[i][2][0]-prev_nodes[k][2][0],2)+math.pow(nodes[i][2][1]-prev_nodes[k][2][1],2)) # 2D distance to check contours
                minax = dd/18
                nx = math.floor(dd/16)
                lineok = True
                circleok = False
                midi_nodes = []
                center = (int((nodes[i][2][0] + prev_nodes[k][2][0])/2), int((nodes[i][2][1] + prev_nodes[k][2][1])/2))
                if nodes[i][2][0] == prev_nodes[k][2][0]:
                    alpha = np.pi/2
                else:
                    alpha = math.atan2(nodes[i][2][1]-prev_nodes[k][2][1],nodes[i][2][0]-prev_nodes[k][2][0])
                versus = 1
                if nx > 0:
                    if nodes[i][2][0] > prev_nodes[k][2][0]:
                        inx = prev_nodes[k][2][0]
                        iny = prev_nodes[k][2][1]
                        dx = (nodes[i][2][0] - prev_nodes[k][2][0])/nx
                        dy = (nodes[i][2][1] - prev_nodes[k][2][1])/nx
                    else:
                        inx = nodes[i][2][0]
                        iny = nodes[i][2][1]
                        dx = (prev_nodes[k][2][0] - nodes[i][2][0])/nx
                        dy = (prev_nodes[k][2][1] - nodes[i][2][1])/nx
                    for iix in range(nx):
                        midi = (int(inx+dx*(iix+1)), int(iny+dy*(iix+1)))
                        midi_nodes.append(midi)
                        # test that points on the line are in the contour, with tolerance
                        found = False
                        for iir in range(-toler, toler+1):
                            for iis in range(-toler, toler+1):
                                a10 = merged_image[midi[1]+iir,midi[0]+iis,0]
                                a11 = merged_image[midi[1]+iir,midi[0]+iis,1]
                                a12 = merged_image[midi[1]+iir,midi[0]+iis,2]
                                if (a10 == 255) and (a11 == 255) and (a12 == 255):
                                # at least one point in the neighbour belongs to the contour
                                    found = True
                                    break
                            if found == True:
                                break
                        if found == False:
                        # NO points of the neighbor belong to the contour: discard the line
                            lineok = False
                            break
                    # look for elliptical lines if the points are not too far from each other
                    #  As the computation for 3D edes is veryu approximate, we do not consider elliptical edges
                    if (lineok == False) and (dd <0):
                        toler = 1
                        for iid in range(1,10):
                            minax = dd/18*iid
                            lineok = True
                            circleok = True
                            midi_nodes = []
                            tmp_nodes = cv2.ellipse2Poly(center,(int(dd/2),int(minax)),int(alpha*180/np.pi),0,180,20)
                            for mmm in tmp_nodes:
                                midi = (mmm[0], mmm[1])
                                midi_nodes.append(midi)
                                found = False
                                for iir in range(-toler, toler+1):
                                    for iis in range(-toler, toler+1):
                                        a10 = merged_image[midi[1]+iir,midi[0]+iis,0]
                                        a11 = merged_image[midi[1]+iir,midi[0]+iis,1]
                                        a12 = merged_image[midi[1]+iir,midi[0]+iis,2]
                                        if (a10 == 255) and (a11 == 255) and (a12 == 255):
                                        # at least one point in the neighbour belongs to the contour
                                            found = True
                                            break
                                    if found == True:
                                        break
                                if found == False:
                                # NO points of the neighbor belong to the contour: discard the line
                                    lineok = False
                                    circleok = False
                                    break
                            if lineok == True:
                                versus = 1
                                break
                            # the other half of the ellipse
                            lineok = True
                            circleok = True
                            midi_nodes = []
                            tmp_nodes = cv2.ellipse2Poly(center,(int(dd/2),int(minax)),int(alpha*180/np.pi),-180,0,20)
                            for mmm in tmp_nodes:
                                midi = (mmm[0], mmm[1])
                                midi_nodes.append(midi)
                                found = False
                                for iir in range(-toler, toler+1):
                                    for iis in range(-toler, toler+1):
                                        a10 = merged_image[midi[1]+iir,midi[0]+iis,0]
                                        a11 = merged_image[midi[1]+iir,midi[0]+iis,1]
                                        a12 = merged_image[midi[1]+iir,midi[0]+iis,2]
                                        if (a10 == 255) and (a11 == 255) and (a12 == 255):
                                        # at least one point in the neighbour belongs to the contour
                                            found = True
                                            break
                                    if found == True:
                                        break
                                if found == False:
                                # NO points of the neighbor belong to the contour: discard the line
                                    lineok = False
                                    circleok = False
                                    break
                            if lineok == True:
                                versus = -1
                                break

                if lineok == True:
        # - lines: [(x,y) of P1, (x,y) of P2, length, index of P1, index of P2, width, [intermediate points], type, paramenters forr ellipses, status]
                    wline = int((prev_nodes[k][3]+nodes[i][3])/2)
                    if circleok == False:
                        type = 'line'
                        leng = math.sqrt(math.pow(nodes[i][2][0]-prev_nodes[k][2][0],2)+math.pow(nodes[i][2][1]-prev_nodes[k][2][1],2)+math.pow(deptz,2)) # 3D distance 
                    else:
                        type = 'ellipse'
                        dd2 = leng = math.sqrt(math.pow(nodes[i][2][0]-prev_nodes[k][2][0],2)+math.pow(nodes[i][2][1]-prev_nodes[k][2][1],2)+math.pow(deptz,2)) 
                        leng = np.pi * ( 3*(dd2/2+minax) - np.sqrt( (3*dd2/2 + minax) * (dd2/2 + 3*minax) ) ) / 2 # approxinate half perimeter
                
                    nodes[i][4].append(prev_nodes[k][0])
                    allnodes[prev_nodes[k][0]-1][4].append(nodes[i][0])
                    # update also the no-triangle list, as no check for triangles will be done
                    nodes[i][5].append(prev_nodes[k][0])
                    allnodes[prev_nodes[k][0]-1][5].append(nodes[i][0])
                    totlines.append({'p1': nodes[i][2], 'p2': prev_nodes[k][2], 'len': leng, 'width': wline, 'ip1glob': nodes[i][0], 'ip2glob': prev_nodes[k][0], 'ip1': i, 'ip2':k, 'midi': midi_nodes, 'type': type, 'center': center, 'maxax': dd/2, 'minax': minax, 'alpha':alpha, 'versus': versus, 'status': 'ok', 'subgraph': 0})

        print("Found %s lines" % len(totlines))

        # draw nodes
        for node in nodes:
            cv2.circle(mergedout, node[2], node[3]+1,(255,0,0),1)
        for node in prev_nodes:
            cv2.circle(mergedout, node[2], node[3]+1,(0,0,255),1)
#        cv2.putText(src8out,str(node[0]),node[2], cv2.FONT_HERSHEY_PLAIN, 1,(0,0,0),1,cv2.LINE_AA) moved later for readability
### skip the triangle check for 3D edges
        # delete from totlines the third edge of triangles if any, setting status=triang
#            for idson in node[4]:
#                if idson < node[0]:
#                # do not consider nodes before the one at hand
#                    continue
#                son = next(nnn for nnn in nodes if nnn[0] == idson)
#                for idgson in son[4]:
#                    if idgson < son[0]:
#                        continue              
#                    if idgson in node[4]:
#                        todelete = next(lll for lll in totlines if (lll['ip1glob'] ==node[0] and lll['ip2glob']==idgson) or (lll['ip2glob'] ==node[0] and lll['ip1glob']==idgson))
#                        todelete['status'] = 'triang'

        # draw edges, markers, etc. Write nodes links without triangles
        last_node = -1
        link_no_tr = []
        for line in totlines:
            if line['status'] == 'triang':
                continue
            if line['type'] == 'line':
                cv2.line(mergedout,line['p1'],line['p2'],(0,255,0),1)
            else:
                if line['versus'] ==1:
                    cv2.ellipse(mergedout,line['center'],(int(line['maxax']),int(line['minax'])),line['alpha']*180/np.pi,0,180,(0,255,0),1)
                else:
                    cv2.ellipse(mergedout,line['center'],(int(line['maxax']),int(line['minax'])),line['alpha']*180/np.pi,-180,0,(0,255,0),1)
            for m in line['midi']:
                cv2.circle(src8out, m, 2,(0,255,255),-1)
## skip triangle check related section
#            nodes[line['ip2']][5].append(line['ip1glob'])
#            if line['ip1'] != last_node:
#                if last_node != -1:
#                    if link_no_tr:
#                        nodes[last_node][5].extend(link_no_tr)
#                last_node = line['ip1']
#                link_no_tr = []
#            link_no_tr.append(line['ip2glob'])
#        if last_node:
#            if link_no_tr:
#                nodes[last_node][5].extend(link_no_tr)

        for node in nodes:
            cv2.putText(mergedout,str(node[0]),node[2], cv2.FONT_HERSHEY_PLAIN, 1,(0,0,0),1,cv2.LINE_AA)
        for node in prev_nodes:
            cv2.putText(mergedout,str(node[0]),node[2], cv2.FONT_HERSHEY_PLAIN, 1,(0,0,0),1,cv2.LINE_AA)
        zeri = "00"
        pp=str(zval-1).strip()
        pp = zeri[:2-len(pp)]+pp
        cv2.imwrite(contourdir+pp+"-"+zname, mergedout)
        alllines.extend(totlines)


    first_img = False
    prev_img = np.copy(src8c)
    prev_nodes = np.copy(nodes)
    allnodes.extend(nodes)

# computation of global structures    
# compute connected subgraphs, and store to nodes and edges. -1 means not connected to anything

print("Total nodes found: %s" % len(allnodes))
print("Total lines found: %s" % len(alllines))
    
subgr = 1
for node in allnodes:
    if not node[5]:
        node[6] = -1
        continue
    if node[6] != 0:
    # already computed from a previous node, copy to linked nodes
        continue
#        nds = []     
#        for nn in node[5]:
#            if nn > node[0]:
#                nds.append(nn)
#        while nds:
#            print("Adding %s nodes" % len(nds))
#            nds1 = []
#            for nn in nds:
#                allnodes[nn-1][6] = node[6]
#                for nn1 in allnodes[nn-1][5]:
#                    if (nn1 > allnodes[nn-1][0] or allnodes[nn1-1][6]!=node[6]) and nn1 not in nds1:
#                        nds1.append(nn1)
#            nds = nds1
            
#        continue
    node[6] = subgr
    print("New subgraph %s" % subgr)
    nds = []     
    for nn in node[5]:
        if nn > node[0]:
            nds.append(nn)
    while nds:
        print("Adding %s nodes" % len(nds))
        nds1 = []
        for nn in nds:
            allnodes[nn-1][6] = node[6]
            for nn1 in allnodes[nn-1][5]:
                if (nn1 > allnodes[nn-1][0] or allnodes[nn1-1][6]!=node[6]) and nn1 not in nds1:
                    nds1.append(nn1)
        nds = nds1
    subgr = subgr+1    

for line in alllines:
    line['subgraph'] = allnodes[line['ip1glob']-1][6]
    

print("Writing nodes and lines data to data/nn.xlt")

workbook = xlwt.Workbook(encoding='utf8')
worksheet = workbook.add_sheet('Nodes')
worksheet.write(1,0,"Index")
worksheet.write(1,1,"Z")
worksheet.write(1,2,"X")
worksheet.write(1,3,"Y")
worksheet.write(1,4,"Width")
worksheet.write(1,5,"Is linked to")
worksheet.write(1,6,"Excluding triangles")
worksheet.write(1,7,"Subgraph")
worksheet.write(1,8,"Type")
worksheet.write(1,9,"El.name")
i = 2
for nn in allnodes:
    worksheet.write(i,0,nn[0])
    worksheet.write(i,1,nn[1])
    worksheet.write(i,2,nn[2][1])
    worksheet.write(i,3,nn[2][0])
    worksheet.write(i,4,nn[3])
    ss = ""
    for nni in nn[4]:
        ss = ss + "%s " % nni
    worksheet.write(i,5,ss)
    ss = ""
    for nni in nn[5]:
        ss = ss + "%s " % nni
    worksheet.write(i,6,ss)
    worksheet.write(i,7,nn[6])
    worksheet.write(i,8,nn[7])
    worksheet.write(i,9,nn[8])
    i = i + 1
    
worksheet = workbook.add_sheet('Edges')
# - lines: [(x,y) of P1, (x,y) of P2, length, index of P1, index of P2, width, [intermediate points], type, paramenters for ellipses, status]
worksheet.write(1,0,"X1")
worksheet.write(1,1,"Y1")
worksheet.write(1,2,"X2")
worksheet.write(1,3,"Y2")
worksheet.write(1,4,"Width")
worksheet.write(1,5,"Length")
worksheet.write(1,6,"Index of P1")
worksheet.write(1,7,"Index of P2")
worksheet.write(1,8,"Type")
worksheet.write(1,9,"Subgraph")
worksheet.write(1,10,"Status")

i = 2
for ll in alllines:
    worksheet.write(i,0,ll['p1'][1])
    worksheet.write(i,1,ll['p1'][0])
    worksheet.write(i,2,ll['p2'][1])
    worksheet.write(i,3,ll['p2'][0])
    worksheet.write(i,4,ll['width'])
    worksheet.write(i,5,ll['len'])
    worksheet.write(i,6,ll['ip1glob'])
    worksheet.write(i,7,ll['ip2glob'])
    worksheet.write(i,8,ll['type'])
    worksheet.write(i,9,ll['subgraph'])
    worksheet.write(i,10,ll['status'])
    i = i + 1
    
workbook.save(datadir+xlsname)

