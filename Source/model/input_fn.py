#######################
# Facies prediction from seismic data by CNN-based semantic segmentation 
# Author: Anshuman Pradhan 
# Email: pradhan1@stanford.edu; pradhan.a269@gmail.com
#######################

# Input function for creating data pipeline from tfrecords files
# Code for decoding tfrecords adapted from examples found online
#########################

import tensorflow as tf

def parse_func(serialized,n_xi,n_yi,n_zi,n_xo,n_yo,n_zo):
    # Define a dict with the data-names and types we expect to
    # find in the TFRecords file.
    # 'seis' refers to input seismic volume
    # 'fac' refers to output facies (rock type class) volume
    features = \
        {
            'seis': tf.FixedLenFeature([], tf.string),
            'fac': tf.FixedLenFeature([], tf.string)
        }

    # Parse the serialized data so we get a dict with our data.
    parsed_example = tf.parse_single_example(serialized=serialized,
                                             features=features)

    # Get the image as raw bytes.
    seis_raw = parsed_example['seis']
    fac_raw = parsed_example['fac']

    # Decode the raw bytes so it becomes a tensor with type.
    seis = tf.cast(tf.reshape(tf.decode_raw(seis_raw, tf.float64),(n_zi,n_xi,n_yi,1)),dtype=tf.float32)
    fac = tf.cast(tf.reshape(tf.decode_raw(fac_raw, tf.float64),(n_zo,n_xo,n_yo,1)),dtype=tf.float32)
    
    return seis, fac



def input_fn(is_training, filenames, params):
    
    # Read in i/p and o/p volume sizes from params files
    n_xi=params.input_size_x
    n_yi=params.input_size_y
    n_zi=params.input_size_z
    n_xo=params.output_size_x
    n_yo=params.output_size_y
    n_zo=params.output_size_z

    if is_training:
        dataset = (tf.data.TFRecordDataset(filenames=filenames)
            .map(lambda f: parse_func(f, n_xi, n_yi, n_zi, n_xo, n_yo, n_zo),num_parallel_calls=params.num_parallel_calls)
            .shuffle(params.train_size)  # whole dataset into the buffer ensures good shuffling
            .batch(params.batch_size)
            .prefetch(1)  # make sure you always have one batch ready to serve
        )
    else:
        dataset = (tf.data.TFRecordDataset(filenames=filenames)
            .map(lambda f: parse_func(f, n_xi, n_yi, n_zi, n_xo, n_yo, n_zo),num_parallel_calls=params.num_parallel_calls)
            .batch(params.batch_size)
            .prefetch(1)  # make sure you always have one batch ready to serve
        )

    # Create reinitializable iterator from dataset
    iterator = dataset.make_initializable_iterator()
    iterator_init_op = iterator.initializer
    seis, fac = iterator.get_next()
    seis

    inputs = {'Seis': seis, 'Facies': fac, 'iterator_init_op': iterator_init_op}
    return inputs