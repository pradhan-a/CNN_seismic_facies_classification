# Reservoir facies (rock type) classification from seismic data using deep 3D CNNs
### Variables & Data
The prediction variable is  facies (rock class) at every pixel of a 3D discretized grid of the petroleum reservoir (see top right image below). In the images below, the plane formed by *x* and *y* dimensions refers to the surface plane of the earth. The *depth* dimension of facies model refers to depth below the earth surface.

The data we have is volumes of seismic data (see top left image below). Note seismic data also has *x* and *y* dimensions, but seismic signals are recorded in time. Hence, the third dimension is *time*.  Note, seismic signals can be recorded along different angles of acquisition, corresponding to *channel* dimension (post/near/far stacks below). Thus, input seismic data is a 4D tensor. Also note, *time* and *depth* dimensions will have, in general, different sizes.

![Alt text](https://github.com/pradhan-a/CNN_rock_type_segmentation/blob/master/Figures/Picture1.png)

### The machine learning problem: use deep 3D CNNs to classify every pixel/voxel of 3D grid of the earth into rock classes given input seismic data (4D tensor)
Our CNN architecture (see image below) is inspired by fully-connected networks used for *semantic segamentation* problems (Long et al., 2015). We use dilated convolutional filters (Yu and Koltun, 2016) to exponentially increase the field of view. Transposed convolutional layers are used to upsample from *time* to *depth* dimension. We also use batch-norm (Ioffe, and Szegedy, 2015) and Monte-Carlo dropout (Gal and Ghahramani, 2015) layers after every convolutional layer. Monte-Carlo dropout calculates the epistemic uncertainty of network.

![Alt text](https://github.com/pradhan-a/CNN_rock_type_segmentation/blob/master/Figures/Picture2.png)

### Tensorflow (tf) impementation
Code [train.py](https://github.com/pradhan-a/CNN_rock_type_segmentation/blob/master/Source/train.py) contains main code for training the CNN. This creates the input data pipeline, builds the tf graph and trains it. We expound on these aspects in detail below.
#### Data processing
* The training data is created by sampling from a prior probability model of geological and geophysical uncertainty (see [Pradhan and Mukerji, 2020](https://github.com/pradhan-a/CNN_rock_type_segmentation/blob/master/References/Pradhan%26Mukerji2020_CNN_seismic_facies.pdf) ). For the example shown in Pradhan and Mukerji, 2020, we had training/validation/test set sizes of 2000/200/200. 
* To efficiently handle the large training data size, we build the tf data pipeline using tfrecords data format. See [build_tfrecords.py](https://github.com/pradhan-a/CNN_rock_type_segmentation/blob/master/Source/build_tfrecord.py) for reading in training examples stored as ascii files and compiling them into a tfrecords file. Once a tfrecords file is created, tf can loads it instantaneously during training.
* [input_fn.py](https://github.com/pradhan-a/CNN_rock_type_segmentation/blob/master/Source/model/input_fn.py) builds the input data pipeline
#### Creating and training the CNN model 
* [model_fn.py](https://github.com/pradhan-a/CNN_rock_type_segmentation/blob/master/Source/model/model_fn.py) builds the tf graph for the CNN shown above. Note for CNN arcitecture shown above, we split the tf graph over two 32 GB Tesla V100 GPUs. 
* We used the softmax crossentropy loss to optimize the network parameters.
#### Evaluating the model
* Use [evaluate.py](https://github.com/pradhan-a/CNN_rock_type_segmentation/blob/master/Source/evaluate.py) to evaluate trained network at test time.
* In [evaluation.py](https://github.com/pradhan-a/CNN_rock_type_segmentation/blob/master/Source/model/evaluation.py), we provide two evaluation functions for cases with and without Monte-Carlo dropout.

### Referencing this work
For citing this work, please use citation  "Pradhan A., and Mukerji, T., Seismic inversion for reservoir facies under geologically realistic prior uncertainty with 3D convolutional neural networks, 90th Annual International Meeting, SEG, Expanded Abstracts, 2020".

### References
* Ioffe, S., Szegedy, C., Batch normalization: accelerating deep network training by reducing internal covariate shift. In: Bach, F., Blei, D. (eds.) Proceedings of the 32nd International Conference on International Conference on Machine Learning - Volume 37(ICML’15), vol. 37, pp. 448–456 (2015) JMLR.org.
* Gal, Y., and Ghahramani, Z., Bayesian convolutional neural networks with Bernoulli approximate variational inference. arXiv:1506.02158, 2015.
* Long, J., Shelhamer, E., Darrell T.,Fully Convolutional Networks for Semantic Segmentation, The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2015, pp. 3431-3440.
* Pradhan A., and Mukerji, T., Seismic inversion for reservoir facies under geologically realistic prior uncertainty with 3D convolutional neural networks, 90th Annual
International Meeting, SEG, Expanded Abstracts, 2020.
* Yu, F., and Koltun, V., Multi-scale context aggregation by dilated convolutions, in Proc. Int. Conf. Learn. Representations, 2016.
