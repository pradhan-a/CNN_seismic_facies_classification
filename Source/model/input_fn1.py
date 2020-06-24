"""Create the input data pipeline using `tf.data`"""

import tensorflow as tf

def _parse_function(seis_file, fac_file, n_x, n_y ):
    """Obtain the image from the filename (for both training and validation).

    The following operations are applied:
        - Decode the image from jpeg format
        - Convert to float and to range [0, 1]
    """
    dseis = tf.data.TextLineDataset(seis_file)
    dseis = dseis.map(lambda f: tf.reshape(tf.decode_csv(f,[[0.0]]*(n_x*n_y)),(n_x,n_y)))

    dfac = tf.data.TextLineDataset(fac_file)
    dfac = dfac.map(lambda f: tf.reshape(tf.decode_csv(f,[[0.0]]*(n_x*n_y)),(n_x,n_y)))


    return dseis, dfac


def input_fn(is_training, seis_filenames, fac_filenames, params):
    """Input function for the SIGNS dataset.

    The filenames have format "{label}_IMG_{id}.jpg".
    For instance: "data_dir/2_IMG_4584.jpg".

    Args:
        is_training: (bool) whether to use the train or test pipeline.
                     At training, we shuffle the data and have multiple epochs
        filenames: (list) filenames of the images, as ["data_dir/{label}_IMG_{id}.jpg"...]
        labels: (list) corresponding list of labels
        params: (Params) contains hyperparameters of the model (ex: `params.num_epochs`)
    """
    num_samples = len(filenames)
    assert len(filenames) == len(labels), "Filenames and labels should have same length"
    n_x=params.image_size_x
    n_y=params.image_size_y
    # Create a Dataset serving batches of images and labels
    parse_fn = lambda f, l: _parse_function(f, l, n_x, n_y)
    if is_training:
		dataset = (tf.data.Dataset.from_tensor_slices((tf.constant(seis_filenames), tf.constant(fac_filenames)))
			.shuffle(num_samples)  # whole dataset into the buffer ensures good shuffling
			.flat_map(parse_fn, num_parallel_calls=params.num_parallel_calls)
			.batch(params.batch_size)
			.prefetch(1)  # make sure you always have one batch ready to serve
			)
	else:
		dataset = (tf.data.Dataset.from_tensor_slices((tf.constant(seis_filenames), tf.constant(fac_filenames)))
			.flat_map(parse_fn)
			.batch(params.batch_size)
			.prefetch(1)  # make sure you always have one batch ready to serve
			)

    # Create reinitializable iterator from dataset
    iterator = dataset.make_initializable_iterator()
    seis, fac = iterator.get_next()
    iterator_init_op = iterator.initializer

    inputs = {'Seis': seis, 'Facies': fac, 'iterator_init_op': iterator_init_op}
    return inputs
