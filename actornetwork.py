import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

from state_representation_network import StateRepresentation

class ActorNetwork(nn.Module):

    def create_network(self, number_inputs, number_actions, hidden_size, window_size, initial_weights):
        self.window_size = window_size
        self.state_rep = StateRepresentation(window_size)
        self.l1 = nn.Linear(number_inputs, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, number_actions)
        self.l3.weight.data.uniform_(-initial_weights, initial_weights)
        self.l3.bias.data.uniform_(-initial_weights, initial_weights)

    def __init__(self, number_inputs, number_actions, hidden_size, window_size, initial_weights=2e-2):
        super(ActorNetwork, self).__init__()
        self.create_network(number_inputs, number_actions, hidden_size, window_size, initial_weights)

    def forward(self, info, rewards):
        state = self.state_rep(info, rewards)
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        x = F.tanh(self.l3(x))
        return state, x

    def get_action(self, info, rewards):
        state, action = self.forward(info, rewards)
        return state, action
