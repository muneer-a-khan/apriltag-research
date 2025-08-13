from frankx import LinearMotion, LinearRelativeMotion, JointMotion, WaypointMotion 
from frankx import Affine, Robot, Waypoint
import math



# Setting up the panda
# robot = Robot("172.16.0.2")
robot = Robot("192.168.1.162")

#robot.set_dynamic_rel(0.05)
robot.velocity_rel = 0.2
robot.acceleration_rel= 0.1
robot.jerk_rel = 0.01 


gripper = robot.get_gripper()
gripper.move(gripper.max_width)
# gripper.clamp()

# Participant positions
front_participant1 = Affine(0.372051, 0.446391, 0.010591, -0.037295, 0.060070, 3.131526)
front_participant2 = Affine(0.589359, -0.476112, 0.021552, -0.590697, -0.003814, -3.138619)

# Thicker audio device(?) properties
speaker_map = [ # 1 = occupied, 0 = not occupied
	[0,0],
	[0,0],
	[0,0]]

component_map = [
	[0,0,0,0],
	[0,0,0,0],
	[0,0,0,0]
]

offset_vertical_speaker = 0.0762 # 3 inches
offset_horizontal_speaker = offset_vertical_speaker * 2 # 6 inches

base_pos_speaker = Affine(0.482111, 0.047504, 0.031231, 0, 0, 3.141592) # base position for audio-speaker things... what even are those??
object_motion_speaker = LinearMotion(base_pos_speaker)

offset_vertical_component = 0.0762
offset_horizontal_component = 0.508 # 2 inches

base_pos_component = Affine(0.489510, -0.256121, 0.023687, 1.57079, 0, 3.141592)
obj_motion_component = LinearMotion(base_pos_component)

m_ideal_pos = JointMotion([0.0, -0.4, 0.0, -1.95, 0.0, 1.6, 0.7])

# Large speaker properties


def place_at_participant(base_pos, offset_horizontal, offset_vertical, pickup_coords, obj_2Dmap, participant_pos):
	x = pickup_coords[0]
	y = pickup_coords[1]
	print(obj_2Dmap)
	if obj_2Dmap[y][x] != 1:
		print("Not occupied")
		return
	obj_2Dmap[y][x] = 0

	obj_vector = Affine(y * offset_vertical, -x * offset_horizontal, 0,0,0,0)
	object_pos = LinearMotion(base_pos * obj_vector)
	dest_pos = LinearMotion(participant_pos)

	robot.move(m_ideal_pos)
	robot.move(object_pos)
	gripper.clamp()
	robot.move(m_ideal_pos)
	robot.move(dest_pos)
	gripper.move(gripper.max_width)
	robot.move(m_ideal_pos)
	return obj_2Dmap

def replace_object(base_pos, offset_horizontal, offset_vertical, dest_coords, obj_2Dmap, participant_pos):
	x_dest = dest_coords[0]
	y_dest = dest_coords[1]
	if obj_2Dmap[y_dest][x_dest] == 1:
		print("Occupied")
		return
	obj_2Dmap[y_dest][x_dest] = 1
	
	dest_vector = Affine(y_dest * offset_vertical, -x_dest * offset_horizontal, 0,0,0,0)

	dest_pos = LinearMotion(base_pos * dest_vector)
	object_pos = LinearMotion(participant_pos)

	#robot.move(m_ideal_pos)
	robot.move(object_pos)
	gripper.clamp()
	# slight vertical movement so the gripper doesn't accidentally move anything
	robot.move(m_ideal_pos)
	robot.move(dest_pos)
	gripper.move(gripper.max_width)
	robot.move(m_ideal_pos)
	return obj_2Dmap

def setup(map):
	if map == speaker_map:
		base_pos = base_pos_speaker
		offset_h = offset_horizontal_speaker
		offset_v = offset_vertical_speaker
	else: # map = component map
		base_pos = base_pos_component
		offset_h = offset_horizontal_component
		offset_v = offset_vertical_component

	for i in range(len(map)):
		for j in range(len(map[0])):
			map = replace_object(base_pos, offset_h, offset_v, (j, i), map, front_participant1)
	
	return map


#speaker_map = setup(speaker_map)
component_map = setup(component_map)

#speaker_map = place_at_participant(base_pos_speaker, offset_horizontal_speaker, offset_vertical_speaker, (0,2), speaker_map, front_participant1)
#speaker_map = replace_object(base_pos_speaker, offset_horizontal_speaker, offset_vertical_speaker, (0,2), speaker_map, front_participant1)
#component_map = place_at_participant(base_pos_component, offset_horizontal_component, offset_vertical_component, (0,0), component_map, front_participant1)
#component_map = replace_object(base_pos_component, offset_horizontal_component, offset_vertical_component, (0,0), component_map, front_participant1)

# Cartesian pose:
    # Pose:  [0.5, 0.0, 0.5, 0.0, 0.0, 0.0]
    # Pose:  [x, y, z, a, b, c]
    # a = end effector rotation around z axis +pi/2 to -pi/2
    # b = end effector rotation around y axis -pi/3 to +pi/4
    # c = end effector rotation around x axis +pi/2 to -pi/2
    # Elbow:  [0.0, -1.0]

# Joints: [0.0, -0.4, 0.0, -1.95, 0.0, 1.6, 0.7]
