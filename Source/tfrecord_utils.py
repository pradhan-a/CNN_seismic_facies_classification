import numpy as np
import tensorflow as tf

def wrap_bytes(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def convert(seis_paths, fac_paths, out_path, params):
    # Args:
    # image_paths   List of file-paths for the images.
    # labels        Class-labels for the images.
    # out_path      File-path for the TFRecords output file.
    
    print("Converting: " + out_path)
    
    # Number of images. Used when printing the progress.
    num_images = len(seis_paths)
    
    # Open a TFRecordWriter for the output-file.
    with tf.python_io.TFRecordWriter(out_path) as writer:
        
        # Iterate over all the image-paths and class-labels.
        for i, (spath, fpath) in enumerate(zip(seis_paths, fac_paths)):
            print("Working on example#"+str(i+1))

            # Load the image-file using matplotlib's imread function.
            seis = np.loadtxt(spath,delimiter=',')
            seis=np.reshape(seis,(params.input_size_z,params.input_size_x,params.input_size_y),order='F')

            fac = np.loadtxt(fpath,delimiter=',')
            fac=np.reshape(fac,(params.output_size_z,params.output_size_x,params.output_size_y),order='F')
            # Convert the image to raw bytes.
            seis_bytes = seis.tostring()
            fac_bytes = fac.tostring()
            # Create a dict with the data we want to save in the
            # TFRecords file. You can add more relevant data here.
            data = \
                {
                    'seis': wrap_bytes(seis_bytes),
                    'fac': wrap_bytes(fac_bytes)
                }

            # Wrap the data as TensorFlow Features.
            feature = tf.train.Features(feature=data)

            # Wrap again as a TensorFlow Example.
            example = tf.train.Example(features=feature)

            # Serialize the data.
            serialized = example.SerializeToString()
            
            # Write the serialized data to the TFRecords file.
            writer.write(serialized)