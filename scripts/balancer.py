#!/usr/bin/env python

import math
import pdb


import numpy as np
import quaternion
import rospy
# from driver import matrix_util
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Quaternion, Twist
from std_msgs.msg import Float64, Bool
import time
import driver.matrix_util as mu

from driver.util import print_teeterbot

from std_srvs.srv import Empty




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

global cmd_vel
cmd_vel = Twist()

global l_wheel_vel
l_wheel_vel = 0.0

global r_wheel_vel
r_wheel_vel = 0.0

# this gets caled each time the state data is updated.
def control_loop(data): 
    q = data.pose[1].orientation
    R = quaternion.as_rotation_matrix(quaternion.from_float_array([q.w, q.x, q.y, q.z]))

    fusedYaw, fusedPitch, fusedRoll, hemi = mu.fused_from_rot_mat(R)

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

def cmd_vel_callback(twist: Twist):
    # print('cmd_vel_callback')
    global cmd_vel
    cmd_vel.linear.x = twist.linear.x
    cmd_vel.linear.y = twist.linear.y
    cmd_vel.linear.z = twist.linear.z
    cmd_vel.angular.x = twist.angular.x
    cmd_vel.angular.y = twist.angular.y
    cmd_vel.angular.y = twist.angular.z
    # pdb.set_trace(header='CALLBACK')

# generic setter
def setter(data, var):
    var = data

def l_wheel_setter(vel):
    global l_wheel_vel
    l_wheel_vel = vel.data
    # print(vel, l_wheel_vel)

def r_wheel_setter(vel):
    global r_wheel_vel
    r_wheel_vel = vel.data
    # print(vel, r_wheel_vel)

if __name__ == '__main__':
    try:
        # reset the model at the very beginning of the simulation
        rospy.wait_for_service('/gazebo/reset_world')
        reset_world = rospy.ServiceProxy('/gazebo/reset_world', Empty)
        reset_world()


        rospy.init_node('balancer', anonymous=True)
        pub_l = rospy.Publisher('teeterbot/left_torque_cmd', Float64, queue_size=1)
        pub_r = rospy.Publisher('teeterbot/right_torque_cmd', Float64, queue_size=1)
        
        # publish a zero command first 
        cmd_teeterbot(0, 0)
        rospy.Subscriber('gazebo/model_states', ModelStates, control_loop)
        rospy.Subscriber('teeterbot/fallen_over', Bool, fallen_over_callback)
        rospy.Subscriber('cmd_vel', Twist, cmd_vel_callback)
        rospy.Subscriber('teeterbot/left_wheel_speed', Float64, l_wheel_setter)
        rospy.Subscriber('teeterbot/right_wheel_speed', Float64, r_wheel_setter)
        # rospy.spin()

        dt = 0.01
        rate = rospy.Rate(1/dt)
        # pdb.set_trace(header='CALLBACK')
        last_pitch = 0
        dpitchdt = 0
        kp = 90; kd = 10
        LIN_VEL_SCALING = 10
        ANG_VEL_SCALING = 0.1
        WHEEL_DAMPING = 0

        while not rospy.is_shutdown():
            if isFallen:
                cmd_teeterbot(0, 0)
                continue
            else:
                # velocity calc
                dpitchdt = (fusedAngles['pitch'] - last_pitch)/dt
                last_pitch = fusedAngles['pitch']

                ## control loop ##
                 
                # pitch regulation
                cmd = kp*fusedAngles['pitch'] + kd*dpitchdt
                # rospy.loginfo("fusedYaw={: 6.4f}, fusedPitch={: 6.4f}, fusedRoll={: 6.4f}, pitchVel={: 6.4f}".format(
                #     fusedAngles['yaw'], fusedAngles['pitch'], fusedAngles['roll'], dpitchdt))
                cmd += LIN_VEL_SCALING*cmd_vel.linear.x 
                rot_cmd = ANG_VEL_SCALING*cmd_vel.angular.y

                # rospy.loginfo(cmd)
                # rospy.loginfo("{}, {}".format(l_wheel_vel, r_wheel_vel))
                
                l_friction = - WHEEL_DAMPING*l_wheel_vel
                r_friction = - WHEEL_DAMPING*r_wheel_vel
                print(l_wheel_vel, r_wheel_vel, l_friction, r_friction)
                
                cmd_l = np.clip(cmd - rot_cmd + l_friction, -10, 10)
                cmd_r = np.clip(cmd + rot_cmd + r_friction, -10, 10)
                # cmd_l = np.clip(cmd - rot_cmd, -10, 10)
                # cmd_r = np.clip(cmd + rot_cmd, -10, 10)
                cmd_teeterbot(cmd_l, cmd_r)
                
                # print_teeterbot("Teeterbot")

            rate.sleep()
        
    except rospy.ROSInterruptException:
        pass
    finally:
        cmd_teeterbot(0, 0)
        
