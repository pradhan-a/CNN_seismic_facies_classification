# Reservoir facies (rock type) classification from seimsic data using deep 3D CNNs
### The classification problem
The prediction variable is  facies (rock class) at every pixel of a 3D discretized grid of the petroleum reservoir (see top right image below). In the images below, the plane formed by *x* and *y* dimensions refer to the surface plane of the earth. The *depth* dimension of facies model refers to depth below the earth surface.

The data we have is volumes of seismic data (see top right image below). Note seismic data also has *x* and *y* dimensions, but seismic signals are recorded in time. Hence, the third dimension is *time*.  Note, seismic signals can be recorded along different angles of acquisition, corresponding to *channel* dimension (post/near/far stacks below). Thus, input seismic data is a 4D tensor. Also note, *time* and *depth* dimensions will have, in general, different sizes.

![Alt text](https://github.com/pradhan-a/CNN_rock_type_segmentation/blob/master/Figures/Picture1.png)

### The Machine learning problem: Learn to classify every pixel/voxel of 3D grid of the earth into rock classes given input seismic data (4D tensor)
Our CNN architecture (see image below) is inspired by fully-connected networks used for *semantic segamentation* problems (Long et al., 2015). We use dilated convolutional filters (Yu and Koltun, 2016) to exponentially increase the field of view. Transposed convolutional layers are used to upsample from *time* to *depth* dimension.

![Alt text](https://github.com/pradhan-a/CNN_rock_type_segmentation/blob/master/Figures/Picture2.png)

### Tensorflow impementation
#### Data processing
The training data is created by sampling from a prior probability model of geological and geophysical uncertainty (see Pradhan and Mukerji, 2020 under "References" directory). For the example shown in Pradhan and Mukerji, 2020, we had training/validation/test set sizes of 2000/200/200. To efficiently handle the large training data size, we build the data pipeline using tfrecords data format. See [code](https://github.com/pradhan-a/CNN_rock_type_segmentation/blob/master/Source/build_tfrecord.py) for reading in training examples stored as ascii files and compiling them into a tfrecords file

### References
* Long, J., Shelhamer, E., Darrell T.,Fully Convolutional Networks for Semantic Segmentation, The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2015, pp. 3431-3440
* Pradhan A., and Mukerji, T., Seismic inversion for reservoir facies under geologically realistic prior uncertainty with 3D convolutional neural networks, 90th Annual
International Meeting, SEG, Expanded Abstracts, 2020.
* Yu, F., and Koltun, V., Multi-scale context aggregation by dilated convolutions, in Proc. Int. Conf. Learn. Representations, 2016.
