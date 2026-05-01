"""
Action space for the web crawling agent.
Actions correspond to selecting the next URL to visit from extracted links.
"""
from dataclasses import dataclass
from typing import List
from .state import State


@dataclass
class Action:
    """
    An action is selecting a URL from the candidate pool.
    
    Attributes:
        state: The state representation of the target URL
        priority: Computed priority score
        reason: Why this action was chosen
    """
    state: State
    priority: float
    reason: str = ""


class ActionSpace:
    """
    Manages the action space - all possible URLs the agent can choose from.
    
    The agent observes the current frontier (unvisited URLs) and selects
    one to visit next based on its scoring policy.
    """
    
    def __init__(self):
        self.candidates: List[Action] = []
        self.action_history: List[Action] = []
    
    def add_candidate(self, state: State, priority: float, reason: str = ""):
        """Add a URL candidate to the action space."""
        action = Action(state=state, priority=priority, reason=reason)
        self.candidates.append(action)
    
    def get_best_action(self, method: str = 'greedy') -> Action:
        """
        Select the best action based on the specified method.
        
        Methods:
            'greedy': Select highest priority action
            'random': Select random action (exploration)
            'epsilon_greedy': Greedy with epsilon probability of random
        """
        if not self.candidates:
            raise ValueError("No candidates in action space")
        
        if method == 'greedy':
            self.candidates.sort(key=lambda x: x.priority, reverse=True)
            action = self.candidates[0]
        
        elif method == 'random':
            import random
            action = random.choice(self.candidates)
        
        elif method == 'epsilon_greedy':
            import random
            epsilon = 0.1
            if random.random() < epsilon:
                action = random.choice(self.candidates)
            else:
                self.candidates.sort(key=lambda x: x.priority, reverse=True)
                action = self.candidates[0]
        else:
            raise ValueError(f"Unknown method: {method}")
        
        self.action_history.append(action)
        return action
    
    def remove_action(self, action: Action):
        """Remove an action from candidates (after visiting)."""
        if action in self.candidates:
            self.candidates.remove(action)
    
    def update_priorities(self, url: str, new_priority: float):
        """Update priority of a specific candidate URL."""
        for action in self.candidates:
            if action.state.url_features.url == url:
                action.priority = new_priority
    
    def get_top_k(self, k: int = 5) -> List[Action]:
        """Get top k actions by priority."""
        sorted_actions = sorted(self.candidates, key=lambda x: x.priority, reverse=True)
        return sorted_actions[:k]
    
    def clear(self):
        """Clear all candidates."""
        self.candidates.clear()
    
    @property
    def size(self) -> int:
        """Number of candidate actions."""
        return len(self.candidates)
