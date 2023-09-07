import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim 
import torch.nn.functional as F

# Hyper parameters
TRANSITION_HISTORY_SIZE = 1000
RECORD_ENEMY_TRANSITIONS = 1.0 # record enemy transitions with probability
BATCH_SIZE = 32
LEARNING_RATE = 0.01
EPSILON = 0.9 # Greedy policy
GAMMA = 0.9 # Discount factor

device = ("cuda"
    if torch.cuda.is_available()
    else "cpu")

class Network(nn.Module):
    """

    """
    def __init__(self):
        super().__init__()

        # Parameters
        self.input_size = 17 * 17
        self.output_size = 6
        # Layers
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(17*17, 121),
            nn.ReLU(),
            nn.Linear(121, 60),
            nn.ReLU(),
            nn.Linear(60, 6),
        )

    def forward(self, x):
        x = torch.Tensor(x).view(1, -1).to(device)
        output = self.linear_relu_stack(x)
        return output

class DQN():
    def __init__(self):

        self.optimizer = optim.Adam
        self.loss_func = nn.MSELoss()

        self.evaluation_network, self.target_network = Network(), Network()
        self.evaluation_network.to(device)
        self.target_network.to(device)



    def learn(self):

        sample_index = np.random.choice(TRANSITION_HISTORY_SIZE, BATCH_SIZE)
        batch = [transitions[i] for i in sample_index]

        batch_state = torch.Tensor([transition.state for transition in batch])
        batch_action = torch.Tensor([transition.action for transition in batch])
        batch_next_state = torch.Tensor([transition.next_state for transition in batch])
        batch_reward = torch.Tensor([transition.reward for transition in batch])

        Q_evaluate = self.evaluation_network(batch_state).gather(1, batch_action)
        Q_next = self.target_network(batch_next_state).detach()
        Q_target = batch_reward + GAMMA * Q_next.max()

        # 
        loss = self.loss_func(Q_evaluate, Q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def choose_action(self, x):
        x = torch.Tensor(x).view(1, -1).to(device)
        if np.random.uniform() < EPSILON:
            actions_value = self.evaluation_network.forward(x).cpu()
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0]
        else:
            action = np.random.randint(0, 6)
        return action