# Driver

Module for controlling the teeterbot (https://github.com/robustify/teeterbot)

# Installation

Developed using ros-noetic (ubuntu 20.04)

Create the catkin workspace as shown in the document.

Navigate to catkin_ws/src and clone the driver module: 
`git clone git@github.com:danielasun/driver.git`

    cd <catkin_ws> 
    mkdir src
    catkin_make
    git clone git@github.com:danielasun/driver.git src/driver
    git clone git@github.com:robustify/teeterbot.git src/teeterbot


# Running the code
In separate terminals which all have `source`d
`source catkin_ws/devel/setup.bash`:

Gazebo/roscore node: `roscd teeterbot_gazebo/launch && roslaunch teeterbot_empty_world.launch`

Keyboard control: `rosrun teleop_twist_keyboard teleop_twist_keyboard.py` 

Controller: `rosrun driver balancer2.py`

You can type in the keyboard control window to control teeterbot using the keys `u i o j k l m , .`.

# How the controller works

I use a cascaded control scheme. There is a low level controller regulating the pitch of the robot using torques and the heading of the robot using a difference in torques. There is also a high level controller which regulates the velocity of the robot by commanding a desired pitch angle.

Some finagling of the state data from gazebo was needed in order to extract the orientation of the robot. I use the fused angles representation of orientation (https://arxiv.org/abs/1809.10105) to rotate the world coordinate information to be back in the body frame.

