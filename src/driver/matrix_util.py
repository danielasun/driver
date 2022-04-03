import math
import numpy as np
from math import acos, atan2, cos, pi, sin, sqrt

def fused_yaw_of_rot_matrix(R):
    """
    Get fused yaw from rotation matrix

    Calculate, wrap and return the fused yaw
    :param R:
    :return:
    """
    trace = R[0, 0]+R[1, 1]+R[2, 2]
    if trace >= 0.0:
        psi_t = atan2(R[1, 0]-R[0, 1], 1+trace)
    elif R[2, 2] >= R[1, 1] and R[2, 2] >= R[0, 0]:
        psi_t = atan2(1.0-R[0, 0]-R[1, 1]+R[2, 2], R[1, 0]-R[0, 1])
    elif R[1, 1] >= R[0, 0]:
        psi_t = atan2(R[2, 1]+R[1, 2], R[0, 2]-R[2, 0])
    else:
        psi_t = atan2(R[0, 2]+R[2, 0], R[2, 1]-R[1, 2])

    fused_yaw = wrap(2*psi_t)
    return fused_yaw


# Conversion: Rotation matrix --> Fused angles (2D)
def fused_pitch_roll_from_rot_mat(R):
    # Calculate the fused pitch and roll
    stheta = -R[2, 0]
    sphi = R[2, 1]

    # Coerce stheta to [-1,1]
    np.clip(stheta, -1, 1)

    # Coerce sphi   to [-1,1]
    np.clip(sphi, -1, 1)
    fusedPitch = math.asin(stheta)
    fusedRoll = math.asin(sphi)

    return fusedPitch, fusedRoll


# Conversion: Rotation matrix --> Fused angles (4D)
def fused_from_rot_mat(R):
    # Calculate the fused yaw, pitch and roll
    fusedYaw = fused_yaw_of_rot_matrix(R)
    fusedPitch, fusedRoll = fused_pitch_roll_from_rot_mat(R)

    # Calculate the hemisphere of the rotation
    hemi = (R[2, 2] >= 0.0)

    return fusedYaw, fusedPitch, fusedRoll, hemi

def wrap(a, b=2*pi):
    """
    wraps a to (-b/2, b/2]
    :param a: val
    :param b: range
    :return: wrapped angle
    """

    # if a is larger than b/2, subtract b from a.
    return a+b*np.floor((b/2-a)/(b))