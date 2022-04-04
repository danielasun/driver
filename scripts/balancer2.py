#!/usr/bin/env python

import math
import pdb


import numpy as np
import quaternion
import rospy
# from driver import matrix_util
from gazebo_msgs.msg import ModelStates, ModelState
from geometry_msgs.msg import Quaternion, Twist
from std_msgs.msg import Float64, Bool
import time
import driver.matrix_util as mu

from driver.util import print_teeterbot
from driver.controller import PIDController

from std_srvs.srv import Empty





class FallenOver(Exception):
    pass
    
pub_l = None
pub_r = None

global fused_angles
fused_angles = {}
fused_angles['yaw'] = 0
fused_angles['pitch'] = 0
fused_angles['roll'] = 0

global isFallen 
isFallen = False

global cmd_vel
cmd_vel = Twist()

global l_wheel_vel
l_wheel_vel = 0.0

global r_wheel_vel
r_wheel_vel = 0.0

global model_states
model_states = ModelStates()

# CALLBACKS

# this gets caled each time the state data is updated.
def model_state_update(data): 
    global model_states
    global fused_angles


    model_states = data
    q = data.pose[1].orientation
    R = quaternion.as_rotation_matrix(quaternion.from_float_array([q.w, q.x, q.y, q.z]))

    fusedYaw, fusedPitch, fusedRoll, hemi = mu.fused_from_rot_mat(R)

    fused_angles['yaw'] = fusedYaw
    fused_angles['pitch'] = fusedPitch
    fused_angles['roll'] = fusedRoll

    v = data.twist[1].linear
    # pdb.set_trace(header='CALLBACK')


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


# processing



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
        rospy.Subscriber('gazebo/model_states', ModelStates, model_state_update)
        rospy.Subscriber('teeterbot/fallen_over', Bool, fallen_over_callback)
        rospy.Subscriber('cmd_vel', Twist, cmd_vel_callback)
        rospy.Subscriber('teeterbot/left_wheel_speed', Float64, l_wheel_setter)
        rospy.Subscriber('teeterbot/right_wheel_speed', Float64, r_wheel_setter)
        # rospy.spin()

        dt = 0.01 # assuming that real time factor is 1.00
        rate = rospy.Rate(1/dt)
        # pdb.set_trace(header='CALLBACK')
        KP_VELOCITY = 0.4
        KD_VELOCITY = 0.25
        KP_PITCH_ANGLE = -90
        KD_PITCH_ANGLE = 10
        KP_ANG_VEL = 1.0
        TEETERBOT_TORQUE_CTRL_MAX = 10

        while not rospy.is_shutdown():
            if isFallen:
                cmd_teeterbot(0, 0)
                continue
            else:
                # velocity calc
                R_world_to_body = mu.r2(fused_angles['yaw']).T

                v_world = model_states.twist[1].linear
                w_world = model_states.twist[1].angular
                v_world = np.array([v_world.x, v_world.y]) 
                w_world = np.array([w_world.x, w_world.y]) 

                # this is approximately true as long as the robot remains upright
                v_body = R_world_to_body@v_world # x component of this is the forwards velocity.
                w_body = R_world_to_body@w_world # y component of this is the rate of change of the pitch

                # pdb.set_trace(header='CALLBACK')

                ##################
                ## control loop ##
                ##################

                """
                Cascaded control loop:
                velocity -> desired pitch
                pitch -> desired torque
                angular velocity -> difference in desired torque
                """

                # velocity regulation through pitch control
                velocity_error = cmd_vel.linear.x - v_body[0] 
                pitch_des = KP_VELOCITY*(velocity_error) - KD_VELOCITY*w_body[1]

                # pitch regulation through torque control
                pitch_error = pitch_des - fused_angles['pitch']
                cmd = KP_PITCH_ANGLE*(pitch_error) + KD_PITCH_ANGLE*(w_body[1])
                
                # angle regulation through torque control 
                ang_vel_error = cmd_vel.angular.y - model_states.twist[1].angular.z
                rot_cmd = KP_ANG_VEL*(ang_vel_error)
                
                torque_cmd_l = np.clip(cmd - rot_cmd, -TEETERBOT_TORQUE_CTRL_MAX, TEETERBOT_TORQUE_CTRL_MAX)
                torque_cmd_r = np.clip(cmd + rot_cmd, -TEETERBOT_TORQUE_CTRL_MAX, TEETERBOT_TORQUE_CTRL_MAX)
                cmd_teeterbot(torque_cmd_l, torque_cmd_r)
                
                print("lin vel cmd:  {: 6.3f} ang vel cmd: {: 6.3f}".format(
                    cmd_vel.linear.x, cmd_vel.angular.y))
                print("lin_vel_body: {: 6.3f} ang_vel_body:{: 6.3f} heading:{: 6.3f}".format(
                    v_body[0], model_states.twist[1].angular.z, fused_angles['yaw']))
                print("desired_pitch:{: 6.3f} torque_cmd_l:{: 6.3f} torque_cmd_r:{: 6.3f}".format(
                    pitch_des, torque_cmd_l, torque_cmd_r))
            rate.sleep()
        
    except rospy.ROSInterruptException:
        pass
    finally:
        cmd_teeterbot(0, 0)
        
