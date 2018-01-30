import numpy as np
import tensorflow as tf

def debugZ(m, z, feed_dict, debug):
    if debug:
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            init.run() # actual init of variable    
            result = z.eval( feed_dict=feed_dict )

        assert ( result.shape == (1,m) )

        def debugExpectedZ():
            return  np.array( [[ 4.,  8., 12., 16., 20., 24., 28.]] )

        assert np.array_equal( result, debugExpectedZ() ) 
        print("debugZ() passed")
