from pyPS4Controller.controller import Controller
from frankx import LinearMotion, LinearRelativeMotion, JointMotion, WaypointMotion
from frankx import Affine, Robot, Waypoint
import frankx
import math
import time
from frankx import Affine, LinearRelativeMotion, robot
import keyboard
import curses
robot = Robot("192.168.1.162")
robot.set_dynamic_rel(0.05)
gripper = robot.get_gripper()
gripper.move(gripper.max_width)
#robot.recover_from_errors()
rest_pose = JointMotion([0.0, -0.4, 0.0, -1.95, 0.0, 1.6, 0.7])
robot.move(rest_pose)
#class MyController(Controller):
	#def __init__(self, **kwargs):
		#Controller.__init__(self, **kwargs)
	#def on_R3_right(self, value):
		#if value < 0:
			#m4 = LinearRelativeMotion(Affine(0.0, 0.1, 0.0))
			#robot.move(m4)
		#else:
			#m4 = LinearRelativeMotion(Affine(0.0,0.0,0.0))
			#robot.move(m4)
	#def on_L3_left(self, value):
		#if value > 0:
			#m4 = LinearRelativeMotion(Affine(0.0, -0.1, 0.0))
			#robot.move(m4)
		#else:
			#m4 = LinearRelativeMotion(Affine(0.0, 0.0, 0.0))
			#robot.move(m4)
	#def on_L3_up(self, value):
		#if value < 0:
			#m4 = LinearRelativeMotion(Affine(0.1, 0.0, 0.0))
			#robot.move(m4)
		#else:
			#m4 = LinearRelativeMotion(Affine(0.0,0.0,0.0))
			#robot.move(m4)
	#def on_L3_down(self, value):
		#if value > 0:
			#m4 = LinearRelativeMotion(Affine(-0.1, 0.0, 0.0))
			#robot.move(m4)
		#else:
			#m4 = LinearRelativeMotion(Affine(0.0,0.0,0.0))
			#robot.move(m4)


#controller = MyController(interface="/dev/input/js0", connecting_using_ds4drv=False)
#controller.listen()
def main(stdscr):
	curses.cbreak()
	stdscr.keypad(True)
	stdscr.nodelay(True)
	m4 = LinearRelativeMotion(Affine(0.0,0.0,0.0))
	key_pressed = False
	while True:
		try:
			k = stdscr.getch()
		except:
			k = -1
	
		if k == curses.KEY_UP:
			if not key_pressed:
				m4 = LinearRelativeMotion(Affine(0.1, 0.0, 0.0))
				stdscr.addstr("you are pressing the up key\n")
				robot.move_async(m4)
				key_pressed = True
		elif k == curses.KEY_DOWN:
			if not key_pressed:
				m4 = LinearRelativeMotion(Affine(-0.1, 0.0, 0.0))
				stdscr.addstr("you are pressing the down key\n")
				robot.move_async(m4)
				key_pressed = True
		elif k == curses.KEY_LEFT:
			if not key_pressed:
				m4 = LinearRelativeMotion(Affine(0.0, 0.1, 0.0))
				stdscr.addstr("you are pressing the left key\n")
				robot.move_async(m4)
				key_pressed = True
		elif k == curses.KEY_RIGHT:
			if not key_pressed:
				m4 = LinearRelativeMotion(Affine(0.0, -0.1, 0.0))
				stdscr.addstr("you are pressing the right key\n")
				robot.move_async(m4)
				key_pressed = True
		else:
			key_pressed = False
			robot.stop_motion()
curses.wrapper(main)



