import json
from jinja2 import Template
from frankx import LinearMotion, LinearRelativeMotion, JointMotion, WaypointMotion 
from frankx import Affine, Robot, Waypoint
import math

# This will now be loaded from task_steps.json
# task_steps = [...]

# Template for code generation:
template = Template("""
{% for i, step in enumerate(steps, start=1) -%}
# Step {{i}}: {{step.desc}}
{% for action in step.actions -%}
{% if action.type == "joint_motion" -%}
joint_motion = JointMotion({{ action.params }})
robot.move(joint_motion)
{% elif action.type == "gripper_clamp" -%}
gripper.clamp()
{% elif action.type == "gripper_release" -%}
gripper.release()
{% endif -%}

{% endfor %}
{% endfor -%}
""")

with open('task_steps.json', 'r') as f:
    task_steps = json.load(f)

output = template.render(steps=task_steps, enumerate=enumerate)

# Setting up the panda
# robot = Robot("172.16.0.2")
robot = Robot("192.168.1.162")

#robot.set_dynamic_rel(0.05)
robot.velocity_rel = 0.7
robot.acceleration_rel= 0.25
robot.jerk_rel = 0.01 


gripper = robot.get_gripper()
gripper.gripper_speed = 0.1
gripper.move(gripper.max_width)
# gripper.clamp()

exec(output)
