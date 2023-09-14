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



    def learn(self):

        if len(self.transitions) < BATCH_SIZE:
            return
        sample_index = np.random.choice(min(len(self.transitions), TRANSITION_HISTORY_SIZE), BATCH_SIZE)

        batch = [self.transitions[i] for i in sample_index]
        #batch_state = [b.state for b in batch]
        #batch_action = [b.action for b in batch]
        #batch_next_state = [b.next_state for b in batch]
        #batch_reward = [b.reward for b in batch]
        #print(batch_next_state)
        #batch_state = torch.tensor(batch_state, dtype=torch.double)
        #batch_action = torch.tensor([INDEX_ACTIONS[i] for i in batch_action], dtype=int).to(device)
        #batch_next_state = torch.tensor(batch_next_state)
        #batch_reward = torch.tensor(batch_reward).to(device)
        Q_evaluates = []
        Q_targets = []
        for transition in batch:
            #print(self.evaluation_network(torch.tensor(transition.state).unsqueeze(0)))
            #print(INDEX_ACTIONS[transition.action])
            #Q_evaluate = self.evaluation_network(torch.tensor(transition.state).unsqueeze(0)).gather(1, torch.tensor(INDEX_ACTIONS[transition.action], device=device).unsqueeze(0))
            Q_evaluate = self.evaluation_network(torch.tensor(transition.state).unsqueeze(0))[0, INDEX_ACTIONS[transition.action]]
            Q_target = transition.reward
            if transition.next_state is not None:
                Q_next = self.target_network(torch.tensor(transition.next_state).unsqueeze(0)).detach()
                Q_target += GAMMA * Q_next.max()
            Q_evaluates.append(Q_evaluate)
            Q_targets.append(Q_target)

        loss = self.loss_func(torch.tensor(Q_evaluates, requires_grad=True), torch.tensor(Q_targets, requires_grad=True))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        #
        #Q_evaluate = self.evaluation_network(batch_state).gather(1, batch_action.unsqueeze(1))
        #Q_next = self.target_network(batch_next_state).detach()
        #Q_target = (batch_reward + GAMMA * Q_next.max())
            
        
        # 


    def get_action(self, x):
        x = torch.Tensor(x).unsqueeze(0).to(device)
        actions_value = self.evaluation_network.forward(x).cpu()
        
        return actions_value