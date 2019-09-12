=======================
Slices in 3D
=======================

-------------------
The problem
-------------------

We have some slices (2D pictures) of a 3D structure made of wires or cylinders that intersect.

We want to reconstruct the 3D plot.

We want also to build a graph with edges, nodes, and width and length of each edge.

Short description of the programs:

- create_contour.py: some basic operations on all the source img

- create_contour_param.py: several operations on a single source img

- create_contour_param2.py: distance transform and watershed with and without Laplace transform on single source img

- create_contour_param3.py: shrinks images and computes lines using Hough method

- create_contour_param4.py: applies distance transform, finds some markers, draw lines on single source image

- create_contour_param4aa.py: same as create_contour_param4.py, but applies threshold without transforming to grayscale


create_contour_param4.py seems the best of these test programs and will be extended to work on all the source images and to compute links between source images too.

- create_all.py; applies the same trnsformations of create_contour_param4.py to all the slices, builds an xls file with the structure (nodes and edges)

- create_contour_electrodes.py: test program to set a grid of circular electrodes onto an image

TBD: merge edges when they run parallel, erase nodes linked to 2 other only, add nodes when edges are merged. One could check that an edge reached a node (e.g. drawing the node in full blue), and stop the edge at this point; check that an edge meets another if there is a "green point" within the threshold, and add a new node there; in a similar way one could check that 2 edges run parallel etc.
