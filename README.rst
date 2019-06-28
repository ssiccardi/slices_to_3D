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

create_contour_param4.py seems the best of these test programs and will be extended to work on all the source images and to compute links between source images too.
