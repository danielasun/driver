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

        dt = 0.01
        rate = rospy.Rate(1/dt)
        # pdb.set_trace(header='CALLBACK')
        last_pitch = 0
        dpitchdt = 0
        KP_ANGLE = 90
        KD_ANGLE = 10
        K_VEL = 0 #100
        LIN_VEL_SCALING = 10
        ANG_VEL_SCALING = 0.1
        WHEEL_DAMPING = 0
        TEETERBOT_TORQUE_CTRL_MAX = 10
        while not rospy.is_shutdown():
            if isFallen:
                cmd_teeterbot(0, 0)
                continue
            else:
                # velocity calc
                dpitchdt = (fused_angles['pitch'] - last_pitch)/dt
                last_pitch = fused_angles['pitch']

                v_world = model_states.twist[1].linear
                v_world = np.array([v_world.x, v_world.y])
                v_body = mu.r2(fused_angles['yaw']).T@v_world # x component of this is the forwards velocity.
                # print(v_body)

                ##################
                ## control loop ##
                ##################

                # pitch regulation #
                cmd = KP_ANGLE*fused_angles['pitch'] + KD_ANGLE*dpitchdt
                # rospy.loginfo("fusedYaw={: 6.4f}, fusedPitch={: 6.4f}, fusedRoll={: 6.4f}, pitchVel={: 6.4f}".format(
                #     fused_angles['yaw'], fused_angles['pitch'], fused_angles['roll'], dpitchdt))
                
                # angle regulation #
                rot_cmd = ANG_VEL_SCALING*cmd_vel.angular.y


                # velocity regulation #
                velocity_error = cmd_vel.linear.x - v_body[0]
                velocity_error = cmd_vel.linear.x
                cmd += LIN_VEL_SCALING*(velocity_error)
                # cmd = cmd + K_VEL*(velocity_error)

                # rospy.loginfo(cmd)
                # rospy.loginfo("{}, {}".format(l_wheel_vel, r_wheel_vel))
                
                l_friction = - WHEEL_DAMPING*l_wheel_vel
                r_friction = - WHEEL_DAMPING*r_wheel_vel
                # print(l_wheel_vel, r_wheel_vel, l_friction, r_friction)
                
                cmd_l = np.clip(cmd - rot_cmd + l_friction, -TEETERBOT_TORQUE_CTRL_MAX, TEETERBOT_TORQUE_CTRL_MAX)
                cmd_r = np.clip(cmd + rot_cmd + r_friction, -TEETERBOT_TORQUE_CTRL_MAX, TEETERBOT_TORQUE_CTRL_MAX)
                # cmd_l = np.clip(cmd - rot_cmd, -10, 10)
                # cmd_r = np.clip(cmd + rot_cmd, -10, 10)
                cmd_teeterbot(cmd_l, cmd_r)
                
                # print_teeterbot("Teeterbot")
                print("{: 6.3f} {: 6.3f} {: 6.3f} {: 6.3f} {: 6.3f} {: 6.3f} {: 6.3f}".format(
                    v_body[0], v_body[1], cmd_l, cmd_r, cmd_vel.linear.x, cmd_vel.linear.x - v_body[0], fused_angles['yaw']))
                
                # rospy.wait_for_service('/gazebo/pause_physics')
                # pause_physics = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
                # pause_physics()
                # pdb.set_trace(header='CALLBACK')
                # rospy.wait_for_service('/gazebo/unpause_physics')
                # unpause_physics = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
                # unpause_physics()


            rate.sleep()
        
    except rospy.ROSInterruptException:
        pass
    finally:
        cmd_teeterbot(0, 0)
        
