import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import pandas as pd
from tqdm import tqdm_notebook as tqdm
import pickle

class CriticNetwork(nn.Module):

	def create_network(self, num_inputs, num_actions, hidden_size, initial_weights):
		self.l1 = nn.Linear(num_inputs + num_actions, hidden_size)
		self.l2 = nn.Linear(hidden_size, hidden_size)
		self.l3 = nn.Linear(hidden_size, 1)
		self.l3.weight.data.uniform_(-initial_weights, initial_weights)
		self.l3.bias.data.uniform_(-initial_weights, initial_weights)

	def __init__(self, num_inputs, num_actions, hidden_size, initial_weights=2e-2):
		super(CriticNetwork, self).__init__()
		self.create_network(num_inputs, num_actions, hidden_size, initial_weights)

	def forward(self, state, action):
		action = torch.squeeze(action)
		x = torch.cat([state, action], 1)
		x = F.relu(self.l1(x))
		x = F.relu(self.l2(x))
		x = self.l3(x)
		return x