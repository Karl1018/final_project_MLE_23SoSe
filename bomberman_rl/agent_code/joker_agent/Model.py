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

INDEX_ACTIONS = {'UP': 0, 'RIGHT': 1, 'DOWN': 2, 'LEFT': 3, 'WAIT': 4, 'BOMB': 5}

device = ("cuda"
    if torch.cuda.is_available()
    else "cpu")

class Network(nn.Module):
    """

    """
    def __init__(self):
        super().__init__()

        # Layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(64 * 9 * 9, 128)
        self.fc2 = nn.Linear(128, 6)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.unsqueeze(1).float().to(device)
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = x.view(batch_size, -1)
        x = nn.functional.relu(self.fc1(x))
        return self.fc2(x)

class DQN():

    def __init__(self):

        self.evaluation_network, self.target_network = Network(), Network()
        self.evaluation_network.to(device)
        self.target_network.to(device)

        self.optimizer = torch.optim.Adam(self.evaluation_network.parameters(), lr=LEARNING_RATE)
        self.loss_func = nn.MSELoss()

    # Computing cumulative discounted future reward
    def calculate_n_step_rewards(transition, n, gamma):
        discount_factor = 1 #第一个R(t)不需要乘以gamma：G = R(t) + γ * R(t+1) + γ^2 * R(t+2) + γ^3 * R(t+3) + ... + γ^(n-1) * R(t+n-1)
        n_step_reward = 0
        n_step_rewards_list = [] 
        for t in range(len(transition)):
            for i in range(n):
                if t + i < len(transition):
                    n_step_reward += discount_factor * transition[t + i].reward
                    discount_factor *= gamma
            n_step_rewards_list.append(n_step_reward)

        return n_step_rewards_list

    def learn(self):

        if len(self.transitions) < BATCH_SIZE:
            return
        sample_index = np.random.choice(min(len(self.transitions), TRANSITION_HISTORY_SIZE), BATCH_SIZE)

        batch = [self.transitions[i] for i in sample_index]
        Q_evaluates = []
        Q_targets = []
        n = 4

        # Calculate n step reward(gamma不知道定多少)你自己传参吧栓Q
        n_step_rewards = calculate_n_step_rewards(self.transitions, n, GAMMA)

        for transition in batch:
            Q_evaluate = self.evaluation_network(torch.tensor(transition.state).unsqueeze(0))[0, INDEX_ACTIONS[transition.action]]
            Q_target = n_step_rewards #改

            if transition.next_state is not None:
                Q_next = self.target_network(torch.tensor(transition.next_state).unsqueeze(0)).detach()
                Q_target += GAMMA * Q_next.max()
            Q_evaluates.append(Q_evaluate)
            Q_targets.append(Q_target)

        loss = self.loss_func(torch.tensor(Q_evaluates, requires_grad=True), torch.tensor(Q_targets, requires_grad=True))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def get_action(self, x):
        x = torch.Tensor(x).unsqueeze(0).to(device)
        actions_value = self.evaluation_network.forward(x).cpu()
        
        return actions_value