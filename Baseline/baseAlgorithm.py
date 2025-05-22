from tqdm import tqdm
import numpy as np
from torch import nn
import gym
import minerl
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import json
import torch.optim as optim
from collections import defaultdict, namedtuple, deque
import random
import copy

class ReplayBuffer:
    def __init__(self, buffer_size, batch_size = 128):
        """
        Initializes the replay buffer.

        Args:
            buffer_size (int): Maximum number of experiences the buffer can hold.
            batch_size (int): Number of experiences to sample per batch.
        """
        self.memory = deque(maxlen=buffer_size)
        self.demo_memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done, isDemo):
        """
        Adds an experience to the replay buffer.

        Args:
            state (Tensor): The current state of the environment.
            action (Tensor): The action taken at this state.
            reward (Tensor): The reward received after taking the action.
            next_state (Tensor): The next state resulting from the action.
            done (bool): Whether the episode has terminated.
        """
        if isDemo:
            self.demo_memory.append((state, action, reward, next_state, done, isDemo))
        else:
            self.memory.append((state, action, reward, next_state, done, isDemo))

    def sample(self, batchSize, isDemo = False):
        """
        Samples a batch of experiences from the replay buffer.

        Returns:
            - state_batch: Batch of states.
            - action_batch: Batch of actions.
            - reward_batch: Batch of rewards.
            - next_state_batch: Batch of next states.
            - done_batch: Batch of terminal state flags.
        """
        if isDemo:
            batch = random.sample(self.demo_memory, batchSize)
        else:
            batch1 = random.sample(self.demo_memory, int( batchSize/2 ))
            batch2 = random.sample(self.memory, int( batchSize/2 ))
            batch = batch1.extend( batch2 )
        # print( batch )
        
        # แยก batch ออกเป็น list ของแต่ละคอลัมน์
        state_batch = [transition[0] for transition in batch]
        action_batch = [transition[1] for transition in batch]
        reward_batch = [transition[2] for transition in batch]
        next_state_batch = [transition[3] for transition in batch]
        done_batch = [transition[4] for transition in batch]
        isDemo = [transition[5] for transition in batch]

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch, isDemo

class BaseAlgorithm():
    """
    Base class for reinforcement learning algorithms.

    Attributes:
        num_of_action (int): Number of discrete actions available.
        action_range (list): Scale for continuous action mapping.
        discretize_state_scale (list): Scale factors for discretizing states.
        lr (float): Learning rate for updates.
        epsilon (float): Initial epsilon value for epsilon-greedy policy.
        epsilon_decay (float): Rate at which epsilon decays.
        final_epsilon (float): Minimum epsilon value allowed.
        discount_factor (float): Discount factor for future rewards.
        q_values (dict): Q-values for state-action pairs.
        n_values (dict): Count of state-action visits (for Monte Carlo method).
        training_error (list): Stores training errors for analysis.
    """

    def __init__(
        self,
        num_of_action: int = 8,
        action_range: list = [-2.0, 2.0],
        learning_rate: float = 1e-3,
        initial_epsilon: float = 1.0,
        epsilon_decay: float = 1e-3,
        final_epsilon: float = 0.001,
        discount_factor: float = 0.95,
        buffer_size: int = 1000,
        batch_size: int = 128,
    ):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.num_of_action = num_of_action
        self.action_range = action_range  # [action_min, action_max]

        self.q_values = defaultdict(lambda: np.zeros(self.num_of_action))
        self.n_values = defaultdict(lambda: np.zeros(self.num_of_action))
        self.training_error = []

        self.w = np.zeros((4, num_of_action))
        self.memory = ReplayBuffer(buffer_size, batch_size)

    
    def decay_epsilon(self):
        """
        Decay epsilon value to reduce exploration over time.
        """
        # ========= put your code here ========= #
        self.epsilon = self.epsilon * self.epsilon_decay
        if self.epsilon <= self.final_epsilon:
            self.epsilon = self.final_epsilon
        pass
        # ====================================== #
    
    def tensor_to_list(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self.tensor_to_list(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.tensor_to_list(v) for v in obj]
        else:
            return obj
        
    def save_value(self, path, filename, value):
        """
        Save weight parameters.
        """
        # ========= put your code here ========= #
        pathFile = path + filename

        value = self.tensor_to_list( value )

        with open( pathFile, "w") as file: 
            json.dump(value, file, indent=4)

    def save_w(self, path, filename):
        """
        Save weight parameters.
        """
        # ========= put your code here ========= #
        pathFile = path + filename
        with open( pathFile, "w") as file: 
            json.dump(self.w.tolist(), file, indent=4)
        # ====================================== #
            
    def load_w(self, path, filename):
        """
        Load weight parameters.
        """
        # ========= put your code here ========= #
        pass
        # ====================================== #
