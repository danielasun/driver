#!/usr/bin/env python

import math
import pdb
from math import acos, atan2, cos, pi, sin, sqrt

import geometry_msgs.msg
import numpy as np
import quaternion
import rospy
# from driver import matrix_util
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Quaternion
from std_msgs.msg import Float64, Bool
import time


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

class FallenOver(Exception):
    pass
    
pub_l = None
pub_r = None

global fusedAngles
fusedAngles = {}
fusedAngles['yaw'] = 0
fusedAngles['pitch'] = 0
fusedAngles['roll'] = 0

global isFallen 
isFallen = False


# this gets caled each time the state data is updated.
def control_loop(data): 
    q = data.pose[1].orientation
    R = quaternion.as_rotation_matrix(quaternion.from_float_array([q.w, q.x, q.y, q.z]))

    fusedYaw, fusedPitch, fusedRoll, hemi = fused_from_rot_mat(R)

    fusedAngles['yaw'] = fusedYaw
    fusedAngles['pitch'] = fusedPitch
    fusedAngles['roll'] = fusedRoll


def fallen_over_callback(fallen):
    if fallen:
        # rospy.loginfo("Fell over")
        isFallen = True
        pub_l.publish(0)
        pub_r.publish(0)
    else:
        isFallen = False
        rospy.loginfo("Reset!")
        # time.sleep(0.1)
        

def cmd_teeterbot(lcmd, rcmd):
    # set both commands to zero if the node is killed
    rospy.logdebug("cmd: {},{}".format(lcmd, rcmd))
    pub_l.publish(lcmd)
    pub_r.publish(rcmd)
    # pdb.set_trace()


if __name__ == '__main__':
    try:
        rospy.init_node('balancer', anonymous=True)
        pub_l = rospy.Publisher('teeterbot/left_torque_cmd', Float64, queue_size=10)
        pub_r = rospy.Publisher('teeterbot/right_torque_cmd', Float64, queue_size=10)
        
        # publish a zero command first 
        cmd_teeterbot(0, 0)
        rospy.Subscriber('gazebo/model_states', ModelStates, control_loop)
        rospy.Subscriber('teeterbot/fallen_over', Bool, fallen_over_callback)
        # rospy.spin()

        dt = 0.01
        rate = rospy.Rate(1/dt) # 10hz 
        # pdb.set_trace(header='CALLBACK')
        last_pitch = 0
        dpitchdt = 0
        kp = 90; kd = 10
        while not rospy.is_shutdown():
            if isFallen:
                cmd_teeterbot(0, 0)
                continue
            else:
                # velocity calc
                dpitchdt = (fusedAngles['pitch'] - last_pitch)/dt
                last_pitch = fusedAngles['pitch']

                # control loop here
                cmd = kp*fusedAngles['pitch'] + kd*dpitchdt
                rospy.loginfo("fusedYaw={: 6.4f}, fusedPitch={: 6.4f}, fusedRoll={: 6.4f}, pitchVel={: 6.4f}".format(
                    fusedAngles['yaw'], fusedAngles['pitch'], fusedAngles['roll'], dpitchdt))
                
                


                # rospy.loginfo(cmd)
                cmd_teeterbot(cmd, cmd)
                # pdb.set_trace(header='CALLBACK')

            rate.sleep()
    except rospy.ROSInterruptException:
        pass
    finally:
        cmd_teeterbot(0, 0)
        
