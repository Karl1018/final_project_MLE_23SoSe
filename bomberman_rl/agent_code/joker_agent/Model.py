import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim 
import torch.nn.functional as F

# Hyper parameters
TRANSITION_HISTORY_SIZE = 10000
RECORD_ENEMY_TRANSITIONS = 1.0 # record enemy transitions with probability
BATCH_SIZE = 150
EPISODE_SIZE = 500
LEARNING_RATE = 0.001
GAMMA = 0.9 # Discount factor
EPSILON_DECAY_DURATION = 10000

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
        x = x.unsqueeze(1).to(device) #TODO: only 1-channel input supported
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

        self.step = 0

        self.optimizer = torch.optim.Adam(self.evaluation_network.parameters(), lr=LEARNING_RATE)
        self.loss_func = nn.MSELoss()
        

    def learn(self):

        if len(self.transitions) < TRANSITION_HISTORY_SIZE:
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
                with torch.no_grad():
                    Q_next = self.target_network(torch.tensor(transition.next_state).unsqueeze(0)).detach()
                Q_target += GAMMA * Q_next.max()
            Q_evaluates.append(Q_evaluate)
            Q_targets.append(Q_target)

        loss = self.loss_func(torch.tensor(Q_evaluates), torch.tensor(Q_targets))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.step += 1
        if self.step % EPISODE_SIZE == 0:
            self.target_network.load_state_dict(self.evaluation_network.state_dict())

    def learn_batched(self):

        if len(self.transitions) < BATCH_SIZE:
            return
        sample_index = np.random.choice(min(len(self.transitions), TRANSITION_HISTORY_SIZE), BATCH_SIZE)
        batch = [self.transitions[i] for i in sample_index]

        states = np.array([sample.state for sample in batch], dtype=np.float32)
        actions = np.array([INDEX_ACTIONS[sample.action] for sample in batch], dtype=np.int64)
        next_states = np.array([sample.next_state for sample in batch], dtype=np.float32)
        rewards = np.array([sample.reward for sample in batch], dtype=np.float32)

        #
        states = torch.tensor(states).to(device)
        actions = torch.tensor(actions).to(device)
        next_states = torch.tensor(next_states).to(device)
        rewards = torch.tensor(rewards).to(device)

        Q_evaluate = self.evaluation_network(states).gather(1, actions.unsqueeze(1))
        #with torch.no_grad():
        Q_next = self.target_network(next_states).detach()
        Q_target = (rewards + GAMMA * Q_next.max()).unsqueeze(1)

        Q_evaluate.requires_grad_(True)
        Q_target.requires_grad_(True)


        loss = self.loss_func(Q_evaluate, Q_target)
        loss.requires_grad_(True)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.step += 1
        if self.step % EPISODE_SIZE == 0:
            self.target_network.load_state_dict(self.evaluation_network.state_dict())


    def get_action(self, x):
        x = torch.Tensor(x).unsqueeze(0).to(device)
        actions_value = self.evaluation_network.forward(x).cpu()
        
        return actions_value
