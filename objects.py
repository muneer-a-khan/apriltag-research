from typing import List, Dict, Any
from enum import Enum 

class Workspace:
    def __init__(self):
        self.inventory: List['Circuit_component'] = []
        self.robot: Robot = Robot()
        self.participant1: Participant = None
        self.participant2: Participant = None

    def get_inventory(self) -> List['Circuit_component']:
        return self.inventory
    def add_to_inventory(self, component: 'Circuit_component'):
        self.inventory.append(component)
    def remove_from_inventory(self, component: 'Circuit_component'):
        self.inventory.remove(component)

    def get_robot(self) -> Robot:
        return self.robot
    def set_robot(self, robot: Robot):
        self.robot = robot

    def get_participant1(self) -> 'Participant':
        return self.participant1
    def set_participant1(self, participant: 'Participant'):
        self.participant1 = participant

    def get_participant2(self) -> 'Participant':
        return self.participant2
    def set_participant2(self, participant: 'Participant'):
        self.participant2 = participant

class Participant:
    def __init__(self, id: int, completed_circuit: 'Circuit'):
        self.id = id
        self.time_spent: float = 0.0
        self.skill: float = 0.0
        self.next_component: 'Circuit_component' = None
        self.current_circuit: 'Circuit' = None
        self.completed_circuit: 'Circuit' = completed_circuit

    def get_id(self) -> int:
        return self.id
    
    def get_time_spent(self) -> float:
        return self.time_spent
    def set_time_spent(self, time: float):
        self.time_spent = time
    
    def get_skill(self) -> float:
        return self.skill
    def set_skill(self, skill: float):
        self.skill = skill

    def get_next_component(self) -> 'Circuit_component':
        return self.next_component
    def set_next_component(self, component: 'Circuit_component'):
        self.next_component = component

    def get_current_circuit(self) -> 'Circuit':
        return self.current_circuit
    def set_current_circuit(self, circuit: 'Circuit'):
        self.current_circuit = circuit

    def get_completed_circuit(self) -> 'Circuit':
        return self.completed_circuit
    def set_completed_circuit(self, circuit: 'Circuit'):
        self.completed_circuit = circuit

    def determine_skill(self) -> float:
        skill = 0.0
        if self.next_component is None:
            return 0.0
        # Placeholder logic for skill determination
        self.skill = skill
        return self.skill
    
    def update_circuit(self, circuit: 'Circuit'):
        self.current_circuit = circuit
        self.next_component = self.determine_next_component()

    def determine_next_component(self) -> 'Circuit_component':        
        if len(self.current_circuit) == 0:
            return self.next_component
        
        for component in self.current_circuit.components:
            # Placeholder logic for determining the next component
            # need to check if component can be attached or if there is no component on the board yet then you can place anything
            continue

        return self.next_component
    
class Base_Action(Enum): # Numbers are determined by how much the action would help the participant
    POINT = 1
    ELABORATE = 2
    PICKUPPLACEITEM = 3

class Action:
    def __init__(self, participant: Participant, base_action: Base_Action, component: 'Circuit_component'):
        self.participant = participant
        self.base_action = base_action
        self.component = component
        self.reward = self.calculate_reward()

    def set_participant(self, participant: Participant):
        self.participant = participant
    def get_participant(self) -> Participant:
        return self.participant
    
    def set_base_action(self, base_action: Base_Action):
        self.base_action = base_action
    def get_base_action(self) -> Base_Action:
        return self.base_action
    
    def set_component(self, component: 'Circuit_component'):
        self.component = component
    def get_component(self) -> 'Circuit_component':
        return self.component
    
    def get_reward(self) -> float:
        return self.reward
    def set_reward(self, reward: float):
        self.reward = reward

    def calculate_reward(self) -> float:
        reward = 0.0
        # Placeholder logic for reward calculation
        self.reward = reward
        return self.reward

class Robot:
    def __init__(self):
        self.queued_action: Action = None

    def get_queued_action(self) -> Action:
        return self.queued_action
    def set_queued_action(self, action: Action):
        self.queued_action = action

    def perform_action(self, action: Action):
        self.queued_action = action
        # Placeholder for actual action logic
        print(f"Performing action: {action.name}")

# placeholder schema for the Circuit graph and nodes from Muneer's code
class Circuit_component:
    def __init__(self, id: int):
        self.id = id

class Circuit:
    def __init__(self):
        self.components: List[Circuit_component] = []
        self.participant: Participant = None
        self.completed_percent: float = 0.0
    
    def get_components(self) -> List[Circuit_component]:
        return self.components
    def add_component(self, component: Circuit_component):
        self.components.append(component)
    def remove_component(self, component: Circuit_component):
        self.components.remove(component)
    
    def get_participant(self) -> Participant:
        return self.participant
    def set_participant(self, participant: Participant):
        self.participant = participant

    def get_completed_percent(self) -> float:
        return self.completed_percent
    def set_completed_percent(self, percent: float):
        self.completed_percent = percent
