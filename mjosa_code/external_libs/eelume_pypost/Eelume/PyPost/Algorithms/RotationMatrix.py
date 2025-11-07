# encoding: utf-8
# ***************************************************************************
# Copyright (C) 2022 Eelume AS - All Rights Reserved
# 
# Unauthorized copying of this file, via any medium is strictly prohibited.
# Proprietary and confidential.
# ***************************************************************************

from scipy.spatial.transform import Rotation as R
import numpy as np

def rotation_matrix(eulerAngles : np.array):
    """
    Generates right hand coordinate system rotation matrix from euler angles.
    
    Parameters
    ----------
    eulerAngles : numpy array
        Euler angles in radians (roll, pitch, yaw)

    Returns
    -------
    Numpy 3x3 matrix
    The rotation matrix
    """
    return R.from_euler('xyz', eulerAngles, degrees=False).as_matrix()
