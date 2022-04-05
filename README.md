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

## Python dependencies
I think you will most likely need to install `numpy` using pip after sourcing `devel/setup.bash`:

    pip install numpy


# Running the code
In separate terminals which all have `source`'d
`source catkin_ws/devel/setup.bash`:

Gazebo/roscore node: `roscd teeterbot_gazebo/launch && roslaunch teeterbot_empty_world.launch`

Keyboard control: `rosrun teleop_twist_keyboard teleop_twist_keyboard.py` 

Controller: `rosrun driver balancer2.py`

Disturbance rejection:
`rosservice call teeterbot/nudge 100 .1`

You can type in the keyboard control window to control teeterbot using the keys 

    u i o       FORWARD & LEFT | MOVE FORWARD | FORWARD & RIGHT
    j k l       ROTATE LEFT    | STOP MOVING  | ROTATE RIGHT
    m , .       BACKWARD & LEFT| MOVE BACKWARD| BACKWARD & RIGHT

**Note:** Sometimes the Controller node can leave gazebo and the teeterbot with large torque commands causing the robot to zoom off the map. I have found the easiest way to reset things to zero was pressing Ctrl+R in the gazebo window while the Controller process is running. Resetting the simulation in this way will move the teeterbot back to the origin and make it upright; you should now be able to restart the controller process.

# How the controller works

I use a cascaded control scheme. (https://www.controleng.com/articles/fundamentals-of-cascade-control/) Each time the control loop runs, the current pitch, linear and angular velocity of the robot are calculated based on callbacks from being subscribed to topics. There is a high level controller which regulates the velocity of the robot by commanding a desired pitch angle. There are a pair of low level controllers regulating the pitch of the robot using torques and the heading of the robot using a difference in commanded torques. Finally, the output of the controller is clamped to be within +/- 10 N-m.

Some finagling of the state data from gazebo was needed in order to extract the orientation of the robot. I use the fused angles representation of orientation (https://arxiv.org/abs/1809.10105) to rotate the world coordinate information to be back in the body frame.

Although a state space model could definitely be used, I usually find it good to start prototyping control solutions using the simplest tools first in order to make sure I have a good grasp on the problem and in this case I believe my solution is satisfactory for a first-order approximation of a solution without further performance or design constraints. In this case, using a set of cascaded PD controllers on velocity and pitch seems to be adequate for achieving a low amount of steady state error in the velocity and pitch.

## Possible improvements to the controller
Further refinements could include model-based control of the robot as an either 2D or 3D and taking over voltage control of the joints. In this case, since I am relatively new to using ROS I focused on ensuring that I had a working implementation of the controller to have a baseline for comparison.


# How the software works
I chose to implement most of my code as a simple node ran from a single script. In this case, because control and state estimation should be synchronized made a lot of sense to just have a single main process running.

The driver module mainly just holds some matrix manipulation code that helps with correctly orienting the robot based on the information from gazebo and teeterbot.

## Possible improvements to the software
A more refined software solution could include the ability to dynamically update the controller gains from the ROS parameter server, modify teeterbot's dynamic properties at runtime using launch files and have abstracted the control loop into more generalizable functions. I would also like to find a way to avoid the use of global variables to retrieve data from ROS callbacks.
