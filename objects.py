from typing import List, Dict, Any
from enum import Enum 

class Workspace:
    def __init__(self):
        self._inventory: List['Circuit_component'] = []  
        self._robot: Robot = Robot()                    
        self._participant1: Participant = None           
        self._participant2: Participant = None           

    def get_inventory(self) -> List['Circuit_component']:
        return self._inventory
    def add_to_inventory(self, component: 'Circuit_component'):
        self._inventory.append(component)
    def remove_from_inventory(self, component: 'Circuit_component'):
        self._inventory.remove(component)

    def get_robot(self) -> 'Robot':
        return self._robot
    def set_robot(self, robot: 'Robot'):
        self._robot = robot

    def get_participant1(self) -> 'Participant':
        return self._participant1
    def set_participant1(self, participant: 'Participant'):
        self._participant1 = participant

    def get_participant2(self) -> 'Participant':
        return self._participant2
    def set_participant2(self, participant: 'Participant'):
        self._participant2 = participant

class Participant:
    def __init__(self, id: int, completed_circuit: 'Circuit'):
        self._id = id                           
        self._time_spent: float = 0.0             
        self._skill: float = 0.0                  
        self._next_component: 'Circuit_component' = None  
        self._current_circuit: 'Circuit' = None   
        self._completed_circuit: 'Circuit' = completed_circuit  

    def get_id(self) -> int:
        return self._id
    
    def get_time_spent(self) -> float:
        return self._time_spent
    def set_time_spent(self, time: float):
        self._time_spent = time
    
    def get_skill(self) -> float:
        return self._skill
    def set_skill(self, skill: float):
        self._skill = skill

    def get_next_component(self) -> 'Circuit_component':
        return self._next_component
    def set_next_component(self, component: 'Circuit_component'):
        self._next_component = component

    def get_current_circuit(self) -> 'Circuit':
        return self._current_circuit
    def set_current_circuit(self, circuit: 'Circuit'):
        self._current_circuit = circuit

    def get_completed_circuit(self) -> 'Circuit':
        return self._completed_circuit
    def set_completed_circuit(self, circuit: 'Circuit'):
        self._completed_circuit = circuit

    def determine_skill(self) -> float:
        skill = 0.0
        if self._next_component is None:
            return 0.0
        # Placeholder logic for skill determination
        self._skill = skill
        return self._skill
    
    def update_circuit(self, circuit: 'Circuit'):
        self._current_circuit = circuit
        self._next_component = self.determine_next_component()

    def determine_next_component(self) -> 'Circuit_component':        
        if self._current_circuit is None or len(self._current_circuit.components) == 0:
            return self._next_component
        
        for component in self._current_circuit.components:
            # Placeholder logic for determining the next component
            # need to check if component can be attached or if there is no component on the board yet then you can place anything
            continue

        return self._next_component

class Base_Action(Enum): # Numbers are determined by how much the action would help the participant
    POINT = 1
    ELABORATE = 2
    PICKUPPLACEITEM = 3

class Action:
    def __init__(self, participant: Participant, base_action: Base_Action, component: 'Circuit_component'):
        self._participant = participant         
        self._base_action = base_action          
        self._component = component              
        self._reward = self.calculate_reward()   

    def set_participant(self, participant: Participant):
        self._participant = participant
    def get_participant(self) -> Participant:
        return self._participant
    
    def set_base_action(self, base_action: Base_Action):
        self._base_action = base_action
    def get_base_action(self) -> Base_Action:
        return self._base_action
    
    def set_component(self, component: 'Circuit_component'):
        self._component = component
    def get_component(self) -> 'Circuit_component':
        return self._component
    
    def get_reward(self) -> float:
        return self._reward
    def set_reward(self, reward: float):
        self._reward = reward

    def calculate_reward(self) -> float:
        reward = 0.0
        skill = self._participant.get_skill()
        value = self._base_action.value
        
        # Placeholder logic for reward calculation
        self._reward = reward
        return self._reward

class Robot:
    def __init__(self):
        self._queued_action: Action = None  # changed public variable to private

    def get_queued_action(self) -> Action:
        return self._queued_action
    def set_queued_action(self, action: Action):
        self._queued_action = action

    def perform_action(self, action: Action):
        self._queued_action = action
        # Placeholder for actual action logic
        print(f"Performing action: {action.get_base_action().name}")

# placeholder schema for the Circuit graph and nodes from Muneer's code
class Circuit_component:
    def __init__(self, id: int):
        self._id = id  

    def get_id(self) -> int:
        return self._id

class Circuit:
    def __init__(self):
        self._components: List[Circuit_component] = [] 
        self._participant: Participant = None           
        self._completed_percent: float = 0.0              
    
    def get_components(self) -> List[Circuit_component]:
        return self._components
    def add_component(self, component: Circuit_component):
        self._components.append(component)
    def remove_component(self, component: Circuit_component):
        self._components.remove(component)
    
    def get_participant(self) -> Participant:
        return self._participant
    def set_participant(self, participant: Participant):
        self._participant = participant

    def get_completed_percent(self) -> float:
        return self._completed_percent
    def set_completed_percent(self, percent: float):
        self._completed_percent = percent