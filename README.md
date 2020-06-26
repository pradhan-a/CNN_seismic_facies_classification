# Reservoir facies classification from seimsic data using deep 3D CNNs
## The tensorflow implementation for estimating/inferring reservoir facies (rock type) from seimsic data

The prediction variable is  facies (rock class) at every pixel of a 3D discretized grid of the petroleum reservoir (see top right image below). In the images below, the plane formed by *x* and *y* dimensions refer to the surface plane of the earth. The *depth* dimension of facies model refers to depth below the earth surface.

The data we have is volumes of seismic data (see top right image below). Note seismic data also has *x* and *y* dimensions, but seismic signals are recorded in time. Hence, the third dimension is *time*.  Note, seismic signals can be recorded along different angles of acquisition, corresponding to *channel* dimension. Thus input seismic data is a 4D tensor. Also note, *time* and *depth* dimensions will have, in general, different sizes.

![Alt text](https://github.com/pradhan-a/CNN_rock_type_segmentation/blob/master/Figures/Picture1.png)
## Machine learnng problem: Learn to classify every pixel of 3D grid of the earth into rock type given input seismic data (4D tensor)
