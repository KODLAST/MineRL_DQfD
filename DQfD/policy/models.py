import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

# Registry for model builders
mapping = dict()

def register(name):
    def _thunk(func):
        mapping[name] = func
        return func
    return _thunk

def get_network_builder(name):
    if callable(name):
        return name
    elif name in mapping:
        return mapping[name]
    else:
        raise ValueError('Registered networks: ' + ', '.join(mapping.keys()))

# MLP base module
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[64, 64], activation=nn.Tanh):
        super().__init__()
        layers = []
        last_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(last_dim, h))
            layers.append(activation())
            last_dim = h
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# Dueling network head
class DuelingMLP(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim=512):
        super().__init__()
        self.advantage = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        self.value = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        a = self.advantage(x)
        v = self.value(x)
        q = v + a - a.mean(dim=1, keepdim=True)
        return q

# Flat DQFD model
class FlatDQFDModel(nn.Module):
    def __init__(self, obs_space, action_space):
        super().__init__()
        input_dim = obs_space.shape[0]
        self.feature_net = MLP(input_dim)
        self.head = DuelingMLP(64, action_space.n)

    def forward(self, obs_dict):
        x = obs_dict['features']
        x = self.feature_net(x)
        return self.head(x)

# NatureCNN for image-based input
class NatureCNN(nn.Module):
    def __init__(self, input_shape, output_dim):
        super().__init__()
        n_input_channels = input_shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            n_flatten = self.cnn(torch.zeros(1, *input_shape)).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))

# CNN-based DQFD model using NatureCNN
class CNN_DQFDModel(nn.Module):
    def __init__(self, obs_space, action_space):
        super().__init__()
        input_shape = (obs_space.shape[2], obs_space.shape[0], obs_space.shape[1])  # (C, H, W)
        self.feature_net = NatureCNN(input_shape, 512)
        self.head = DuelingMLP(512, action_space.n)

    def forward(self, obs_dict):
        obs_dict = torch.tensor(obs_dict, dtype=torch.float32)  # Convert NumPy array to PyTorch tensor
        obs_dict = obs_dict.permute(0, 3, 1, 2)/255.0
        x = self.feature_net(obs_dict)
        return self.head(x)

# Register models
@register("flat_dqfd")
def make_flat_model(name, obs_space, action_space, reg=1e-5):
    return FlatDQFDModel(obs_space, action_space)

@register("minerl_dqfd")
def make_cnn_model(name, obs_space, action_space, reg=1e-5):
    return CNN_DQFDModel(obs_space, action_space)

@register("tf1_minerl_dqfd")
def make_tf1_compat_model(name, obs_space, action_space, reg=1e-5):
    return CNN_DQFDModel(obs_space, action_space)


