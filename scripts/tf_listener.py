#!/usr/bin/env python  
import rospy
import pdb

import math
from math import acos
from math import atan2
from math import cos
from math import pi
from math import sin
from math import sqrt
import geometry_msgs.msg
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Quaternion
import quaternion
import numpy as np
from driver import matrix_util


model_data = {'ground_plane': 0, 
              'teeterbot': 1
              }

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


def callback(data):
    # rospy.loginfo(rospy.get_caller_id() + "data = {}".format(data))
    # print(data)

    # pos = data.pose[1].
    q = data.pose[1].orientation
    R = quaternion.as_rotation_matrix(quaternion.from_float_array([q.w, q.x, q.y, q.z]))
    # print(R)

    fusedYaw, fusedPitch, fusedRoll, hemi = fused_from_rot_mat(R)
    print(fused_from_rot_mat(R))

    # pdb.set_trace(header='CALLBACK')


def listener():
    print('init listener')
    rospy.init_node('state_listener', anonymous=True)
    # joint_state_ = rospy.Subscriber('teeterbot/joint_states', ModelStates, callback)
    model_state_listener = rospy.Subscriber('gazebo/model_states', ModelStates, callback)

    # rate = rospy.Rate(10.0)
    # while not rospy.is_shutdown():
    rospy.spin()

        

if __name__ == '__main__':
    listener()
    
    # tfBuffer = tf2_ros.Buffer()
    # listener = tf2_ros.TransformListener(tfBuffer)

    # rospy.wait_for_service('spawn')
    # spawner = rospy.ServiceProxy('spawn', turtlesim.srv.Spawn)
    # turtle_name = rospy.get_param('turtle', 'turtle2')
    # spawner(4, 2, 0, turtle_name)

    # turtle_vel = rospy.Publisher('%s/cmd_vel' % turtle_name, geometry_msgs.msg.Twist, queue_size=1)

    # rate = rospy.Rate(10.0)
    # while not rospy.is_shutdown():
    #     try:
    #         trans = tfBuffer.lookup_transform(turtle_name, 'turtle1', rospy.Time())
    #     except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
    #         rate.sleep()
    #         continue

    #     msg = geometry_msgs.msg.Twist()

    #     msg.angular.z = 4 * math.atan2(trans.transform.translation.y, trans.transform.translation.x)
    #     msg.linear.x = 0.5 * math.sqrt(trans.transform.translation.x ** 2 + trans.transform.translation.y ** 2)

    #     turtle_vel.publish(msg)

    #     rate.sleep()