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
target_pose = JointMotion([1.2045596475363218, 0.2194290079003886, -0.026285752144560477, -3.032488093618516, 0.1485544076567339, 3.3027833802435134, 1.8299664947669236])
destination_pose = JointMotion([0.8398623717814161, 0.7200363930233737, -0.3404285875906941, -1.9418959081716207, 0.2448649704406932, 2.6380285274834963, 1.1326621188700519])
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


