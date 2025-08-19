from typing import List, Dict, Any
from enum import Enum 

class Participant:
    def __init__(self, id: int):
        self.id = id
        self.time_spent: float = 0.0
        self.skill: float = 0.0
        self.next_component: 'Circuit_component' = None
        self.circuit: 'Circuit' = None

    def determine_skill(self) -> float:
        """
        Determines the skill level of the participant based on how long they take
        """
        skill = 0.0
        if self.next_component is None:
            return 0.0
        # Placeholder logic for skill determination
        self.skill = skill
        return self.skill
    
    def update_circuit(self, circuit: 'Circuit'):
        self.circuit = circuit
        self.next_component = self.determine_next_component()

    def determine_next_component(self) -> 'Circuit_component':
        """
        Determines the next component for the participant to work on
        """
        if self.next_component is None:
            return None
        # Placeholder logic for determining the next component
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

class Robot:
    def __init__(self):
        self.queued_action: Action = None
        self.participants: List[Participant] = []

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

    def add_component(self, component: Circuit_component):
        self.components.append(component)
