import math
import logging
from math import acos
from math import atan2
from math import cos
from math import pi
from math import sin
from math import sqrt

import numpy as np
from numba import float64
from numba import njit

import pebl.math.modern_robotics as mr
from pebl.math.modern_robotics import TransToRp

log = logging.getLogger(__name__)

def return_shapeless(vec_function):
    """
    decorator function which will reshape a vector output from (1,n) or (n,1) to (n,).
    has no effect if one of the dimensions is not 1

    :param vec_function: function for which to reshape output
    :return: return of vec_function as a shapeless vector
    """

    def wrapper(*args, **kwargs):

        v_shaped = vec_function(*args, **kwargs)

        try:
            if v_shaped.shape[0] == 1:
                v_shapeless = np.array(v_shaped[0])
            elif v_shaped.shape[1] == 1:
                v_shapeless = np.array(v_shaped.T[0])
            else:
                v_shapeless = v_shaped
        except IndexError:
            v_shapeless = v_shaped

        return v_shapeless

    return wrapper


@njit(cache=True)
def nb_block(X):
    xtmp1 = np.hstack(X[0])
    xtmp2 = np.hstack(X[1])
    return np.vstack((xtmp1, xtmp2))


@njit(float64[:, :](float64), cache=True)
def rx(q):
    c = cos(q)
    s = sin(q)
    return np.array(((1, 0, 0), (0, c, -s), (0, s, c)))


@njit(float64[:, :](float64), cache=True)
def ry(q):
    c = cos(q)
    s = sin(q)
    return np.array(((c, 0, s), (0, 1, 0), (-s, 0, c)))


@njit(float64[:, :](float64), cache=True)
def rz(q):
    c = cos(q)
    s = sin(q)
    return np.array(((c, -s, 0), (s, c, 0), (0, 0, 1)))


@njit(float64[:, :](float64, float64, float64), cache=True)
def rpy(roll, pitch, yaw):
    return np.dot(rz(yaw), np.dot(ry(pitch), rx(roll)))


@njit(float64[:, :](float64, float64, float64), cache=True)
def xyz_euler(roll, pitch, yaw):
    return np.dot(rx(roll), np.dot(ry(pitch), rz(yaw)))


@njit(float64[:, :](float64, float64, float64), cache=True)
def abg_to_R(a, b, g):
    return np.dot(rx(a), np.dot(ry(b), rz(g)))


@njit(float64[:, :](float64, float64, float64), cache=True)
def rpy_to_unit_vector(roll, pitch, yaw):
    return np.array((
        (sin(pitch),),
        (-cos(pitch)*sin(roll),),
        (cos(pitch)*cos(roll),),
    ))


@njit(float64[:, :](float64[:]), cache=True)
def skew(omg):
    """

    :param omg: Takes a 3-vector (angular velocity).
    :return: Returns the skew symmetric matrix in so3.

    Example Input:
        omg = [1, 2, 3]
        Output:
        [[ 0, -3,  2],
         [ 3,  0, -1],
         [-2,  1,  0]]

    """

    return np.array(((0, -omg[2], omg[1]), (omg[2], 0, -omg[0]), (-omg[1],
                                                                  omg[0], 0)))


# @njit(float64[:,:](float64[:,:], float64[:,:]), cache=True)
def transform(R=np.eye(3), p=np.c_[0, 0, 0].T):
    # creates transform using np.array
    A = np.zeros((4, 4))
    A[:3, :3] = R
    A[:3, 3:] = p
    A[3, 3] = 1
    return A


@njit(float64[:, :](float64[:, :]), cache=True)
def invtransform(T):
    A = np.zeros((4, 4))
    R = T[:3, :3]
    A[:3, :3] = R.T
    A[:3, 3] = -R.T@T[:3, 3]
    A[3, 3] = 1
    return A


@njit(float64[:, :](float64, float64, float64), cache=True)
def zyz_euler_angle(a, b, g):
    """
    Creates a rotation matrix based on z-y-z rotation angles

    :param a: 1st z rotation, in radians
    :param b: 2nd y rotation, in radians
    :param g: 3rd z rotation, in radians
    :return: 3x3 rotation matrix
    """
    return np.array(
            ((cos(a)*cos(b)*cos(g)-sin(a)*sin(g),
              -cos(a)*cos(b)*sin(g)-sin(a)*cos(g), cos(a)*sin(b)),
             (sin(a)*cos(b)*cos(g)+cos(a)*sin(g),
              -sin(a)*cos(b)*sin(g)+cos(a)*cos(g), sin(a)*sin(b)),
             (-sin(b)*cos(g), sin(b)*sin(g), cos(b))))


@njit(cache=True)
def inverse_zyz_euler_angle(R, flip_beta=True):
    """
    Given a rotation matrix R, find the corresponding ZYZ Euler angle decomposition

    :param R: 3x3 rotation matrix
    :param flip_beta:
    :return: (yaw, pitch, yaw) rotation tuple
    """

    if flip_beta:
        beta = atan2(-sqrt(R[2, 0]**2+R[2, 1]**2), R[2, 2])
    else:
        beta = atan2(-sqrt(R[2, 0]**2+R[2, 1]**2), R[2, 2])
    if beta == 0 or beta == 2*pi:
        alpha = 0
        gamma = atan2(-R[0, 1], R[0, 0])

    elif beta == pi or beta == -pi:
        alpha = 0
        gamma = atan2(R[0, 1], -R[0, 0])

    else:
        alpha = atan2(R[1, 2]/sin(beta), R[0, 2]/sin(beta))
        gamma = atan2(R[2, 1]/sin(beta), -R[2, 0]/sin(beta))
    return alpha, beta, gamma


@njit(float64[:](float64[:], float64[:]), cache=True)
def minicross3(a, b):
    """
    cross product for only single 3 dimensional vectors
    """
    return np.array([
        a[1]*b[2]-a[2]*b[1],
        a[2]*b[0]-a[0]*b[2],
        a[0]*b[1]-a[1]*b[0]
    ])


@njit(float64[:, :](float64[:, :], float64[:]), cache=True)
def create_transform(R, p):
    """
    Creates a transform without all of the shape issues that the jit function
    has
    :param R: rotation matrix
    :param p: position vector
    :return:
    """
    A = np.zeros((4, 4))
    A[:3, :3] = R
    A[:3, 3] = p
    A[3, 3] = 1
    return A


@njit(float64[:, :](float64[:]), cache=True)
def create_transform_from_position(p):
    return create_transform(np.eye(3), p)


@njit(float64[:, :](float64[:, :]), cache=True)
def create_transform_from_rotation(R):
    return create_transform(R, np.zeros(3))


@njit(float64[:, :](), cache=True)
def create_blank_transform():
    return create_transform(np.eye(3), np.zeros(3))


@njit(float64[:, :](float64), cache=True)
def r2(a=None):
    """
    2x2 rotation matrix
    :param a: angle in radians
    :return: 2x2 rotation matrix
    """

    s = np.sin(a)
    c = np.cos(a)
    return np.array([[c, -s], [s, c]])


@njit(float64(float64[:, :]), cache=True)
def get_yaw_from_rotation_matrix(R):
    return atan2(R[1, 0], R[0, 0])


'''
Spatial Dynamics
'''

REVOLUTE = 0
PRISMATIC = 1


# @njit(cache=True)
def jcalc(joint_type, q):
    """
    Calculates the joint transform X, motion subspace S and constraint force subspace T
    :param joint_type: joint type (revolute or prismatic)

    :param q: joint coordinate
    :return: X, S, T
    """

    if joint_type == REVOLUTE:
        # print("REVOLUTE")
        R = rz(q)
        p = np.array(((0,), (0,), (0,)), np.float64)
        S = np.array(((0,), (0,), (1,), (0,), (0,), (0,)))
        T = np.array(((1, 0, 0, 0, 0), (0, 1, 0, 0, 0), (0, 0, 0, 0, 0),
                      (0, 0, 1, 0, 0), (0, 0, 0, 1, 0), (0, 0, 0, 0, 1)))
    elif joint_type == PRISMATIC:
        # print("PRISMATIC")
        R = np.eye(3)
        p = np.array((
            (0,),
            (0,),
            (q,),
        ), np.float64)
        S = np.array(((0,), (0,), (0,), (0,), (0,), (1,)))
        T = np.array(((1, 0, 0, 0, 0), (0, 1, 0, 0, 0), (0, 0, 1, 0, 0),
                      (0, 0, 0, 1, 0), (0, 0, 0, 0, 1), (0, 0, 0, 0, 0)))
    else:
        raise TypeError(
                'Unrecognized joint type, must be 0 (REVOLUTE) or 1 (PRISMATIC)')

    return transform(R, p), S, T


@njit
def calcVWerr(Tref, Tnow):
    Rref, pref = TransToRp(Tref)
    Rnow, pnow = TransToRp(Tnow)

    perr = pref-pnow
    Rerr = Rnow.T@Rref
    werr = (Rnow@rot2omega(Rerr).T).T[0]

    return perr, werr


@njit(float64[:, :](float64[:, :]), cache=True)
def rot2omega(R):
    """
    Transform rotation matrix into the corresponding angular velocity vector
    ref. Introduction To Humanoid Robotics by Kajita (2013) pg 67

    :param R:
    :return:
    """

    el = np.array((
        (R[2, 1]-R[1, 2],),
        (R[0, 2]-R[2, 0],),
        (R[1, 0]-R[0, 1],),
    )).T
    norm_el = np.linalg.norm(el)
    A = np.zeros((1, 3))
    # print el, norm_el
    if norm_el > 1e-6:  # normal case
        # print "normal case"
        B = np.arctan2(norm_el, np.trace(R)-1)/norm_el*el
        A[0] = B[0]
    elif R[0, 0] > 0 and R[1, 1] > 0 and R[2, 2] > 0:  # identity matrix
        # print "identity"
        C = np.array((
            (0,),
            (0,),
            (0,),
        )).T
        A[0] = C[0]
    else:
        # print "singularity"
        D = np.pi/2*np.array(((R[0, 0]+1,), (R[1, 1]+1,),
                              (R[2, 2]+1,)), ).T
        A[0] = D[0]

    return A


@njit(float64[:](float64[:, :], float64[:, :]), cache=True)
def calcWerr(Rref, Rnow):
    # calculates the angular rotation error between two rotation matrices
    Rerr = Rnow.T@Rref
    werr = (Rnow@rot2omega(Rerr).T).T[0]

    return werr


@njit(float64[:, :](float64[:], float64), cache=True)
def rodrigues(w, t):
    """
    Integrate an angular velocity for t seconds or rotate about a unit rotation axis
    for t radians using Rodrigues' formula.

    :param w: angular velocity
    :param t: time (if using non-unit vector w), angle (in radians) if using a unit rotation axis
    :return: 3x3 integrated rotation matrix
    """

    w_x = np.zeros((3, 3))
    A = mr.VecToso3(w)
    w_x[0] = A[0]
    w_x[1] = A[1]
    w_x[2] = A[2]
    ewdt = np.eye(3)+np.sin(t)*w_x+(1-np.cos(t))*np.dot(w_x, w_x)
    return ewdt


@njit(float64[:, :](float64[:, :]), cache=True)
def ortho_regularize(R):
    """
    Regularizes rotation matrices or other orthogonal matrices using svd to keep them
    from accumulating numerical errors. Takes about .006 ms per call.

    :param R: matrix to be regularized.
    :return: R, an orthonormalized matrix.
    """

    U, S, V = np.linalg.svd(R)
    return U@V


'''
dynamics
'''


@njit(float64[:, :](float64[:]), cache=True)
def skew(x):
    return np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])


@njit(float64[:, :](float64, float64[:, :], float64[:]), cache=True)
def parallel_axis_theorem(mass, inertia, r):
    """
    Generalized parallel axis theorem in 3D
    Reference: https://aapt.scitation.org/doi/full/10.1119/1.4994835
    :param mass: mass of link
    :param inertia: 3x3 inertia tensor
    :param r: 3-vector from arbitrary point to the center of mass
    :return: modified inertia matrix about the origin of the vector r
    """
    return inertia+mass*np.array(
            [[r[1]**2+r[2]**2, -r[0]*r[1], -r[0]*r[2]],
             [-r[0]*r[1], r[0]**2+r[2]**2, -r[1]*r[2]],
             [-r[0]*r[2], -r[1]*r[2], r[0]**2+r[1]**2]])


# @njit(float64[:,:](float64[:,:], float64[:]), cache=True)
def create_adjoint_transform(R=None, p=None):
    """
    Adjoint representation of a homogenous transform.
    :param R: 3x3 rotation matrix
    :param p: 3-vector position
    :return: adjoint matrix
    """
    if R is None:
        R = np.eye(3)
    if p is None:
        p = np.zeros(3)

    return np.block([[R, skew(p)], [np.zeros((3, 3)), R]])


# @njit(float64[:,:](float64[:,:], float64[:]), cache=True)
def adjoint_from_transform(T):
    return create_adjoint_transform(R=T[:3, :3], p=T[:3, 3])


# @njit(float64[:,:](float64[:,:]), cache=True)
def adjoint_inverse(adj):
    """
    mathematically get inverse of adjoint transform T so that T@T_inv = I

    :param adj:
    :return: T inverse
    """
    Rt = adj[:3, :3].T

    return np.block([[Rt, -Rt@adj[:3, 3:]@Rt], [np.zeros((3, 3)), Rt]])


# @njit
def extended_rotation_matrix(R):
    """

    :param R: a 3x3 rotation matrix
    :return:
    """

    return np.block([[R, np.zeros((3, 3))], [np.zeros((3, 3)), R]])


@njit(float64[:](float64[:, :], float64[:]), cache=True)
def transform_multiply(T, p):
    return T[:3, :3]@p+T[:3, 3]


def tilt_angle_to_tilt_phase(p, g, a):
    """

    :param y: fused yaw
    :param g: gamma, angle of tilt axis
    :param a: alpha, angle of tilt
    :return:
    """
    px = a*np.cos(g)
    py = a*np.sin(g)
    pz = p

    return np.array([px, py, pz])


def tilt_angles_to_rot(angles):
    p, g, a = angles
    c = cos
    s = sin
    R = np.array([
        [c(g)*c(g+p)+c(a)*s(g)*s(g+p), s(g)*c(g+p)-c(a)*c(g)*s(g+p),
         s(a)*s(g+p)],
        [c(g)*s(g+p)-c(a)*s(g)*c(g+p), s(g)*s(g+p)+c(a)*c(g)*c(g+p),
         -s(a)*c(g+p)],
        [-s(a)*s(g), s(a)*c(g), c(a)]])
    return R


def tilt_phase_to_tilt_angles(p):
    """

    :param p: p is a 3-vector Px, Py, Pz
    :return:
    """
    alpha = np.sqrt(p[0]**2+p[1]**2)
    gamma = atan2(p[1], p[0])
    psi = p[2]
    return (psi, gamma, alpha)


def tilt_angles_to_tilt_phase(angles):
    """
    convert tilt angles to tilt-phase parameters
    :param angles: [psi, gamma, alpha]
    :return:
    """
    p, g, a = angles
    px = a*np.cos(g)
    py = a*np.sin(g)
    pz = p
    return np.array([px, py, pz])


def tilt_phase_to_rot(p):
    return tilt_angles_to_rot(tilt_phase_to_tilt_angles(p))


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


def wrap(a, b=2*pi):
    """
    wraps a to (-b/2, b/2]
    :param a: val
    :param b: range
    :return: wrapped angle
    """

    # if a is larger than b/2, subtract b from a.
    return a+b*np.floor((b/2-a)/(b))

def rot_to_tilt_angles(R):
    psi = fused_yaw_of_rot_matrix(R)
    gamma = atan2(-R[2, 0], R[2, 1])
    alpha = acos(np.clip(R[2, 2], -1, 1))

    return [psi, gamma, alpha]


def rot_to_tilt_phase(R):
    tilt_angles = rot_to_tilt_angles(R)
    return tilt_angles_to_tilt_phase(tilt_angles)


def sign(x):
    return math.copysign(1, x)


def rot_to_fused(R):
    fusedYaw = fused_yaw_of_rot_matrix(R)
    fusedPitch = math.asin(-R[2, 0])
    fusedRoll = math.asin(R[2, 1])
    hemi = (R[2, 2] >= 0.0)
    return fusedYaw, fusedPitch, fusedRoll, hemi


def fused_angles_to_rot(fusedYaw, fusedPitch, fusedRoll,
                        hemi: bool) -> np.array:
    """

    :param fusedYaw:
    :param fusedPitch:
    :param fusedRoll:
    :param hemi: True/False
    :return:
    """

    # Precalculate the sine values
    sth = sin(fusedPitch)
    sphi = sin(fusedRoll)

    # Calculate the sine sum criterion
    crit = sth*sth+sphi*sphi

    # Calculate the tilt angle alpha
    if crit >= 1.0:
        calpha = 0.0
        salpha = 1.0
    else:
        if hemi:
            calpha = sqrt(1.0-crit)
        else:
            calpha = -sqrt(1.0-crit)
        salpha = sqrt(crit)

    # Calculate the tilt axis angle gamma
    gamma = atan2(sth, sphi)

    # Precalculate terms involved in the rotation matrix expression
    cgam = cos(gamma)
    sgam = sin(gamma)
    psigam = fusedYaw+gamma
    cpsigam = cos(psigam)
    spsigam = sin(psigam)
    A = cgam*cpsigam
    B = sgam*cpsigam
    C = cgam*spsigam
    D = sgam*spsigam

    # Calculate and return the required rotation matrix
    R = np.array([[A+D*calpha, B-C*calpha, salpha*spsigam],
                  [C-B*calpha, D+A*calpha, -salpha*cpsigam],
                  [-sth, sphi, calpha]])
    return R


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


def rot_to_ypr(R):
    """
    Z-Y-X Euler angles

    Note: since euler angles are not unique, this may not be a smooth conversion
    :param R: 3x3 orientation matrix
    :return: yaw, pitch, roll euler angles
    """
    yaw = atan2(R[1,0], R[0,0])
    pitch = atan2(-R[2, 0], sqrt(R[2, 1]**2+ R[2,2]**2))
    roll = atan2(R[2, 1], R[2,2])
    return yaw, pitch, roll

def antirodrigues(a, b):
    a = a/np.linalg.norm(a)
    b = b/np.linalg.norm(b)
    v = minicross3(a, b)
    c = np.dot(a, b)
    return np.eye(3) + skew(v) + skew(v)@skew(v)*(1/(1+c))
