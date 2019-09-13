# this program reads two csv files, nodes.csv and edges.csv, corresponding to the 2 sheets of the xls output of the create_all.py program
# it considers only nodes and edges of a subgraph, and after trianlge are cleaned
# it deletes nodes connected to other 2 only, and the edges joining them; it creates instead a single edge between the 2 exteremes
# i.e. before: 1---2---3 after 1------3
# moreover, if a node is joined just to another, that is if it has a free extreme, it sets conventionally = 0 the starting point of the edge
#
# the output is a single csv with both nodes and edges, in a format that will be used to compute the potentials

import csv
import io
#from cStringIO import StringIO
from optparse import OptionParser

parser = OptionParser()
parser.add_option("-N", "--nimage", dest="im_elect",
                  help="Choose Image Number holding electrodes")
(optlist, args) = parser.parse_args()

if not optlist.im_elect:
	errmsg='This programs needs the number of the image holding electrodes as a command line argument'
	raise SyntaxError(errmsg)
else:
    im_elect = optlist.im_elect


subgraph = "1"  # the connected subgraph that we are interested in
nodes = list(csv.reader(open('data/nodes_'+im_elect+'.csv'), delimiter=","))
edges = list(csv.reader(open('data/edges_'+im_elect+'.csv'), delimiter=","))

mynodes = []
indnodes = []
myedges = {}

loading = False
nnodes = 0
# loading structure for nodes: id, z, x, y, nodes linked, flag to consider the node or not
for node in nodes:
    if loading == False:
        if node[0] == 'Index':
            loading = True
        continue
    if node[7].strip() != subgraph:
        continue
    mynodes.append([node[0],node[1],node[2],node[3],node[6],True,node[8]])
    indnodes.append(node[0])
    nnodes = nnodes + 1

print("I consider %s nodes" % nnodes)

loading = False
nedges = 0
# loading structure for edges: key (for easy of retrieval): p1-p2, data [p1, p2, lenght], flag to consider the node or not
for edge in edges:
    if loading == False:
        if edge[0] == 'X1':
            loading = True
        continue
    if edge[10].strip() != 'ok':
        continue
    if edge[9].strip() != subgraph:
        continue
    key = edge[6].strip()+'-'+edge[7].strip()
    l_edge = 0
    try:
        l_edge = float(edge[5])
    except:
        print("Lenght error for edge %s - %s" % (edge[6],edge[7]))
        pass
    l_width = 0
    try:
        l_width = float(edge[4])
    except:
        print("Widht error for edge %s - %s" % (edge[6],edge[7]))
        pass
    myedges[key] = [edge[6],edge[7],l_edge,True,l_width]
    nedges = nedges + 1

print("I consider %s edges" % nedges)

for node in mynodes:
    linked = node[4].split()
    if len(linked) == 1:
        continue # skip this
    # if the node is linked just to another, it is a free end of its edge, we are not interested in it and we flag the edge setting its starting point to zero
        key = str(node[0]).strip()+'-'+node[4].strip()
        if key in myedges:
            myedges[key][0] = 0
        else:
            key = node[4].strip()+'-'+str(node[0]).strip()
            if key in myedges:
                myedges[key][1] = 0
            else:
                print("edge not found between %s and %s" % (node[0],node[4]))
        continue
    if len(linked) == 2:
        continue  # skip the following, as we ant to keep also this kind of nodes, that may appear e.g. in loops
        node[5] = False  # DON'T consider this node
        l_edge = 0
        key = str(node[0]).strip()+'-'+linked[0].strip()
        # DON'T consider the edge leading to the first linked node
        if key in myedges:
            l_edge = l_edge + myedges[key][2]
            myedges[key][3] = False
        else:
            key = linked[0].strip()+'-'+str(node[0]).strip()
            if key in myedges:
                l_edge = l_edge + myedges[key][2]
                myedges[key][3] = False
        key = str(node[0]).strip()+'-'+linked[1].strip()
        # DON'T consider the edge leading to the second linked node
        if key in myedges:
            l_edge = l_edge + myedges[key][2]
            myedges[key][3] = False
        else:
            key = linked[1].strip()+'-'+str(node[0]).strip()
            if key in myedges:
                l_edge = l_edge + myedges[key][2]
                myedges[key][3] = False
        # ADD an edge between the linked nodes, if they are not yet linked (should never happen)
        key = linked[0].strip()+'-'+linked[1].strip()
        key1 = linked[1].strip()+'-'+linked[0].strip()
        if not key in myedges and not key1 in myedges:
            if linked[0].strip()<linked[1].strip():
                myedges[key]=[linked[0].strip(),linked[1].strip(),l_edge,True]
            else:
                myedges[key1]=[linked[1].strip(),linked[0].strip(),l_edge,True]
        # remove links to the deleted node nad replace with mutual links between the 2 joined nodes
        pp1 = indnodes.index(linked[0].strip())
        ll1=mynodes[pp1][4].split()
        pp2=ll1.index(node[0])
        ll1.pop(pp2)
        ll1.append(linked[1])
        ss = ""
        for s in ll1:
            ss = ss + s + " "
        mynodes[pp1][4] = ss.strip()

        pp1 = indnodes.index(linked[1].strip())
        ll1=mynodes[pp1][4].split()
        pp2=ll1.index(node[0])
        ll1.pop(pp2)
        ll1.append(linked[0])
        ss = ""
        for s in ll1:
            ss = ss + s + " "
        mynodes[pp1][4] = ss.strip()


nnodes = 0
buf=io.StringIO()
writer=csv.writer(buf, quoting=csv.QUOTE_NONNUMERIC, delimiter=",")
writer.writerow(['id', 'z', 'x', 'y', 'nodes linked', 'type'])

for node in mynodes:
    if node[5] == True:
        writer.writerow([node[0],node[1],node[2],node[3],node[4],node[6]])
        nnodes = nnodes+1

#out=base64.encodestring(buf.getvalue())
text_file = open("data/nodes_ok_"+im_elect+".csv", "w")
text_file.write(buf.getvalue())
text_file.close()


print("I am saving %s nodes" % nnodes)

nedges = 0
buf1=io.StringIO()
writer1=csv.writer(buf1, quoting=csv.QUOTE_NONNUMERIC, delimiter=",")
writer1.writerow(['id', 'P1', 'P2', 'length','width'])

for key in myedges:
    edge = myedges[key]
    if edge[3] == True:
        writer1.writerow([key,edge[0],edge[1],edge[2],edge[4]])
        nedges = nedges+1

text_file1 = open("data/edges_ok_"+im_elect+".csv", "w")
text_file1.write(buf1.getvalue())
text_file1.close()

print("I am saving %s edges" % nedges)
