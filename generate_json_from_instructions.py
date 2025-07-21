import json

task_steps_data = [
    	{"desc": "Pick and place object", "actions": [
        {"type": "joint_motion", "params": [1.1397136980546145, 0.3829155423306582, -0.07622295587104662, -2.6355864172806402, -0.005245058147443563, 3.0413267635219556, 1.8981889969209829]},
        {"type": "gripper_clamp"},
        {"type": "joint_motion", "params": [0.0, -0.4, 0.0, -1.95, 0.0, 1.6, 0.7]},
	{"type": "joint_motion", "params": [-0.640485113266728, 0.644795482184577, 0.33760805421544793, -2.1254269037702773, -0.4554432689863499, 2.7484092197234915, 0.8939128440801399]},
        {"type": "gripper_release"}
    ]},
    # Add more steps with different actions here
]

with open('task_steps.json', 'w') as f:
    json.dump(task_steps_data, f, indent=4)

print("task_steps.json created successfully.")
