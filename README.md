# CellSeg

![Banner](doc/image/banner.png)


A generalist algorithm for cell segmentation from 3D labeled image. 

## Segmentation
Segmentation part generate 7 csv files, that can be populated later during the analysis part.
- __cell_plane_df.csv__ contains measure relative to the cell, such as volume, number of neighbours, orientation, curvature
- __cell_plane_df.csv__ contains plane measure, such as orientation, anisotropy, area, perimeter
- __edge_df.csv__ contains measure relative to the edge such as length (real & shortest), curvature, the 3 connected cells
- __edge_pixel_df.csv__ contains pixels coordinates of each edge and the 3 connected cells
- __face_df.csv__ contains coordinates of the middle of the face, the lengths, the angles
- __face_edge_pixel_df.csv__ contains coordinates of the edges associated to the face, and the coordinates of the middle of the face
- __face_pixel_df.csv__ contains pixels coordinates of each face and the 2 connected cells and edges

![coordinate](doc/image/coordinate.png)




## Install 

See [INSTALL.md](INSTALL.md) for a step by step install. 