import time
from frankx import Affine, Robot


robot = Robot("192.168.1.162")
gripper = robot.get_gripper()


state = robot.read_once()
print('\nPose: ', robot.current_pose())
print('O_TT_E: ', state.O_T_EE)
print('Joints: ', state.q)
print('Elbow: ', state.elbow)
