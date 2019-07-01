import numpy as np
import cv2
from optparse import OptionParser
import math
import random as rng
import io

import xlwt

######################
#
#
# Elaboration:
# - reads input file and extracts z value = nn looking for _z0nn_
# - finds contours
# - applies distance transform, saved as gray/nn.jpg
# - applies threshold to dst transformed, saved as thresh/nn.jpg
# - finds markers and lines joining them, saved as contourdir/nn.jpg,orig_nn.jpg
#
# Creates structures:
# - nodes: [index, z, (x,y), width, [linked to], [linked - triangles], subgraph]
# - lines: [(x,y) of P1, (x,y) of P2, length, index of P1, index of P2, width, [intermediate points], type, paramenters forr ellipses, status, subgraph]
# Width of a node = radius of circle centerd at the node and contained in the contour
# Widh of a line = average width of the nodes it joins
# Parameters
# -I image name in slices directory
# -T threshold value (default = 80



threshold_value = 80   # can be adjusted depending on acutal images

slicedir = "Zslices/"
graydir = "gray/"
threshdir = "thresh/"
contourdir = "contour/"
datadir = "data/"
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
if not "_z0" in fname:
    errmsg = "Image name must contain z value in the form _z0nn_"
    raise SyntaxError(errmsg)

xxx = fname.find("_z0")
zname = fname[xxx+3:xxx+5]+".jpg"
xlsname = fname[xxx+3:xxx+5]+".xls"
zval = int(fname[xxx+3:xxx+5])
print("Outpur saved as "+zname)


print("Finding contours")
img = cv2.imread(slicedir+fname+'.png')
kernel = np.ones((5,5),np.float32)/25
src = cv2.filter2D(img,-1,kernel)

imgray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

#ret, thresh2 = cv2.threshold(imgray, threshold_value, 255, cv2.THRESH_BINARY_INV)

thresh2 = np.zeros((src.shape[0], src.shape[1]), np.uint8)
thresh2[:,:] = 255
for i in range(1024):
    for k in range(1024):
        if (src[i][k][0] >19) and (src[i][k][1] >19) and(src[i][k][2] >200):
            thresh2[i][k] = 0
            
cv2.imwrite(contourdir+'aa_thr_'+zname, thresh2)
            
contours2, hierarchy2 = cv2.findContours(thresh2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#print(len(contours2))

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

cv2.imwrite(contourdir+'aa_cont_'+zname, blank_image2)


# distance transform
bw1 = cv2.cvtColor(blank_image2, cv2.COLOR_BGR2GRAY)
_, bw1 = cv2.threshold(bw1, threshold_value, 255, cv2.THRESH_BINARY)# | cv2.THRESH_OTSU)



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
#cv2.imwrite(contourdir+'example11.jpg', markers1)


print("Drawing circles in markers (contour/nn.jpg, contour/orig_nn.jpg)")
#print(contours1[0])
#src8c = cv2.imread(contourdir+'example8.jpg')
src8c = blank_image2
src8out = np.copy(blank_image2)
src8out[src8out<10] = 102  # gray backgroud for easier reading

nodes = []
# - nodes: [index, z, (x,y), width, [linked to],[linked - triangles]]
indnode = 1
# store nodes
for i in range(len(contours1)):
    mask = np.zeros((markers1.shape[0], markers1.shape[1]), dtype=np.ubyte)
    cv2.drawContours(mask,contours1,i,1,cv2.FILLED)
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(dist1a, mask)
    #cv2.drawContours(src, contours1, i, (255,255,255), -1)
#    cv2.circle(src, maxLoc, int(maxVal)+1,(0,255,0),1)
#    cv2.circle(src8out, maxLoc, int(maxVal)+1,(255,0,0),1)
    nodes.append([indnode,zval, maxLoc, int(maxVal), [], [],0])
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
#                midi = (int(iny+dy*iix),int(inx+dx*iix))
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
            # llok for elliptical lines if the points are not too far from each other
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
            wline = int((nodes[k][3]+nodes[i][3])/2)
            if circleok == False:
                type = 'line'
                leng = dd  # distance of the 2 points
                #cv2.line(src8out,nodes[i][2],nodes[k][2],(0,0,255),2)
                #cv2.line(src,nodes[i][2],nodes[k][2],(0,255,0),2)
            else:
                type = 'ellipse'
                leng = np.pi * ( 3*(dd/2+minax) - np.sqrt( (3*dd/2 + minax) * (dd/2 + 3*minax) ) ) / 2 # approxinate half perimeter
                #cv2.ellipse(src8out,center,(int(dd/2),int(minax)),alpha*180/np.pi,0,180,(0,0,255),2)
                #cv2.ellipse(src,center,(int(dd/2),int(minax)),alpha*180/np.pi,0,180,(0,255,0),2)
            
            nodes[i][4].append(nodes[k][0])
            nodes[k][4].append(nodes[i][0])
#            for m in midi_nodes:
#                cv2.circle(src8out, m, 2,(0,255,0),-1)
#                cv2.circle(src, m, 2,(0,255,0),-1)
            totlines.append({'p1': nodes[i][2], 'p2': nodes[k][2], 'len': leng, 'width': wline, 'ip1': nodes[i][0], 'ip2': nodes[k][0], 'midi': midi_nodes, 'type': type, 'center': center, 'maxax': dd/2, 'minax': minax, 'alpha':alpha, 'versus': versus, 'status': 'ok', 'subgraph': 0})
print("Found %s lines" % len(totlines))

# draw nodes
for node in nodes:
    cv2.circle(src, node[2], node[3]+1,(0,255,0),1)
    cv2.circle(src8out, node[2], node[3]+1,(255,0,0),1)
    cv2.putText(src8out,str(node[0]),node[2], cv2.FONT_HERSHEY_PLAIN, 1,(0,0,0),1,cv2.LINE_AA)
# delete from totlines the third edge of triangles if any, setting status=triang
    for idson in node[4]:
        if idson < node[0]:
        # do not consider nodes before the one at hand
            continue
        son = next(nnn for nnn in nodes if nnn[0] == idson)
        for idgson in son[4]:
            if idgson < son[0]:
                continue              
            if idgson in node[4]:
                todelete = next(lll for lll in totlines if (lll['ip1'] ==node[0] and lll['ip2']==idgson) or (lll['ip2'] ==node[0] and lll['ip1']==idgson))
                todelete['status'] = 'triang'

# draw edges, markers, etc. Write nodes links without triangles
last_node = -1
link_no_tr = []
for line in totlines:
    if line['status'] == 'triang':
        continue
    if line['type'] == 'line':
        cv2.line(src8out,line['p1'],line['p2'],(0,0,255),2)
        cv2.line(src,line['p1'],line['p2'],(0,255,0),2)
    else:
        if line['versus'] ==1:
            cv2.ellipse(src8out,line['center'],(int(line['maxax']),int(line['minax'])),line['alpha']*180/np.pi,0,180,(0,0,255),2)
            cv2.ellipse(src,line['center'],(int(line['maxax']),int(line['minax'])),line['alpha']*180/np.pi,0,180,(0,255,0),2)
        else:
            cv2.ellipse(src8out,line['center'],(int(line['maxax']),int(line['minax'])),line['alpha']*180/np.pi,-180,0,(0,0,255),2)
            cv2.ellipse(src,line['center'],(int(line['maxax']),int(line['minax'])),line['alpha']*180/np.pi,-180,0,(0,255,0),2)
    for m in line['midi']:
        cv2.circle(src8out, m, 2,(0,255,0),-1)
        cv2.circle(src, m, 2,(0,255,0),-1)
    nodes[line['ip2']-1][5].append(line['ip1'])
    if line['ip1'] != last_node:
        if last_node != -1:
            if link_no_tr:
                nodes[last_node-1][5].extend(link_no_tr)
        last_node = line['ip1']
        link_no_tr = []
    link_no_tr.append(line['ip2'])
if last_node:
    if link_no_tr:
        nodes[last_node-1][5].extend(link_no_tr)
# compute connected subgraphs, and store to nodes and edges. -1 means not connected to anything

    
    
subgr = 1
for node in nodes:
    if not node[5]:
        node[6] = -1
        continue
    if node[6] != 0:
    # already computed from a previous node, copy to linked nodes
        nds = []     
        for nn in node[5]:
            if nn > node[0]:
                nds.append(nn)
        while nds:
            nds1 = []
            for nn in nds:
                nodes[nn-1][6] = node[6]
                for nn1 in nodes[nn-1][5]:
                    if (nn1 > nodes[nn-1][0] or nodes[nn1-1][6]!=node[6]) and nn1 not in nds1:
                        nds1.append(nn1)
            nds = nds1
        continue
    node[6] = subgr
    for nn in node[5]:
        if nn > node[0]:
             nodes[nn-1][6] = node[6]
    subgr = subgr+1    

for line in totlines:
    line['subgraph'] = nodes[line['ip1']-1][6]
    
cv2.imwrite(contourdir+'orig_'+zname, src)
cv2.imwrite(contourdir+zname, src8out)

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
i = 2
for nn in nodes:
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
for ll in totlines:
    worksheet.write(i,0,ll['p1'][1])
    worksheet.write(i,1,ll['p1'][0])
    worksheet.write(i,2,ll['p2'][1])
    worksheet.write(i,3,ll['p2'][0])
    worksheet.write(i,4,ll['width'])
    worksheet.write(i,5,ll['len'])
    worksheet.write(i,6,ll['ip1'])
    worksheet.write(i,7,ll['ip2'])
    worksheet.write(i,8,ll['type'])
    worksheet.write(i,9,ll['subgraph'])
    worksheet.write(i,10,ll['status'])
    i = i + 1
    
#fp = io.StringIO()
workbook.save(datadir+xlsname)
#fp.seek(0)
#fp.close()

