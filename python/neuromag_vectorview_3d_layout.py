"""DecMeg2014 example code.

Simple visualization of the Neuromag VectorView 3D layout.

Copyright Emanuele Olivetti 2014, BSD license, 3 clauses.
"""

import numpy as np
from scipy.io import loadmat
from mayavi import mlab
    
if __name__ == '__main__':

    filename = '../additional_files/NeuroMagSensorsDeviceSpace.mat'
    data = loadmat(filename, chars_as_strings=True)
    position = data['pos']
    orientation = data['ori']
    label = data['lab']
    sensor_type = data['typ']

    # Normalize orientation for visualization purpose:
    orientation = orientation / np.sqrt((orientation * orientation).sum(1))[:,None]
    mlab.quiver3d(position[:,0], position[:,1], position[:,2], orientation[:,0], orientation[:,1], orientation[:,2], color=(1,0,0))
    mlab.show()
