# Reservoir facies (rock type) classification from seimsic data using deep 3D CNNs
### The classification problem
The prediction variable is  facies (rock class) at every pixel of a 3D discretized grid of the petroleum reservoir (see top right image below). In the images below, the plane formed by *x* and *y* dimensions refer to the surface plane of the earth. The *depth* dimension of facies model refers to depth below the earth surface.

The data we have is volumes of seismic data (see top right image below). Note seismic data also has *x* and *y* dimensions, but seismic signals are recorded in time. Hence, the third dimension is *time*.  Note, seismic signals can be recorded along different angles of acquisition, corresponding to *channel* dimension (post/near/far stacks below). Thus, input seismic data is a 4D tensor. Also note, *time* and *depth* dimensions will have, in general, different sizes.

![Alt text](https://github.com/pradhan-a/CNN_rock_type_segmentation/blob/master/Figures/Picture1.png)

### The Machine learning problem: Learn to classify every pixel/voxel of 3D grid of the earth into rock classes given input seismic data (4D tensor)
Our CNN architecture (see image below) is inspired by fully-connected networks used for *semantic segamentation* problems (Long et al., 2015). We use dilated convolutional filters (Yu and Koltun, 2016) 

![Alt text](https://github.com/pradhan-a/CNN_rock_type_segmentation/blob/master/Figures/Picture2.png)

### References
* Long, J., Shelhamer, E., Darrell T.,Fully Convolutional Networks for Semantic Segmentation, The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2015, pp. 3431-3440
* Yu, F., and Koltun, V., Multi-scale context aggregation by dilated convolutions, in Proc. Int. Conf. Learn. Representations, 2016.
