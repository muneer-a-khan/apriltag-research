from frankx import LinearMotion, LinearRelativeMotion, JointMotion, WaypointMotion 
from frankx import Affine, Robot, Waypoint
import math



# Setting up the panda
# robot = Robot("172.16.0.2")
robot = Robot("192.168.1.162")

#robot.set_dynamic_rel(0.05)
robot.velocity_rel = 0.3
robot.acceleration_rel= 0.1 
robot.jerk_rel = 0.01 


# gripper = robot.get_gripper()
# gripper.release(0.084)
# gripper.clamp()

gripper = robot.get_gripper()
gripper.move(gripper.max_width)
 
m_ideal_pos = JointMotion([0.0, -0.4, 0.0, -1.95, 0.0, 1.6, 0.7])

# m_ideal_pos = LinearMotion(Affine(0.5, 0.0, 0.5, 0.0, 0.0, 0.0))

# m_ideal_pos = LinearMotion(Affine(0.5, 0.0, 0.5, 0.0, 0.0, 0.0),elbow = 0.0)

robot.move(m_ideal_pos)


# Cartesian pose:
    # Pose:  [0.5, 0.0, 0.5, 0.0, 0.0, 0.0]
    # Pose:  [x, y, z, a, b, c]
    # a = end effector rotation around z axis +pi/2 to -pi/2
    # b = end effector rotation around y axis -pi/3 to +pi/4
    # c = end effector rotation around x axis +pi/2 to -pi/2
    # Elbow:  [0.0, -1.0]

# Joints: [0.0, -0.4, 0.0, -1.95, 0.0, 1.6, 0.7]
