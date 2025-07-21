from frankx import LinearMotion, LinearRelativeMotion, JointMotion, WaypointMotion
from frankx import Affine, Robot, Waypoint
import frankx
import math
import time

robot = Robot("192.168.1.162")
robot.set_dynamic_rel(0.05)
gripper = robot.get_gripper()
gripper.move(gripper.max_width)
#robot.recover_from_errors()
rest_pose = JointMotion([0.0, -0.4, 0.0, -1.95, 0.0, 1.6, 0.7])
target_pose = JointMotion([1.7830494866617, 0.5613594803491176, -0.7552694055869811, -2.6732520261731123, -0.025457418718271785, 3.158195185248538, 1.9090048748354087])
destination_pose = JointMotion([1.1481295773376496, 0.8644083585676722, -0.6352532002298455, -2.0473319385556867, -0.023600970592338393, 3.1071499900814694, 1.6547671865161484])
robot.move(rest_pose)
robot.move(target_pose)
gripper.clamp()
robot.move(destination_pose)
gripper.move(gripper.max_width)
robot.move(rest_pose)
#while True:
#	time.sleep(0.5)
#	state = robot.read_once()
#	print(state.q)
