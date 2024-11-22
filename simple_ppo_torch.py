
import torch as T
import torch.nn as nn
import torch.optim as optim
import numpy as np

class ActorNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.fc3 = nn.Linear(fc2_dims, n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        state = state.view(state.size(0), -1)  # Flatten the 2D input
        x = T.relu(self.fc1(state))
        x = T.relu(self.fc2(x))
        actions = T.softmax(self.fc3(x), dim=-1)
        return actions

class CriticNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.fc3 = nn.Linear(fc2_dims, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        state = state.view(state.size(0), -1)  # Flatten the 2D input
        x = T.relu(self.fc1(state))
        x = T.relu(self.fc2(x))
        value = self.fc3(x)
        return value

class Agent:
    def __init__(self, actor_lr, critic_lr, input_dims, n_actions, fc1_dims=128, fc2_dims=128, gamma=0.99, epsilon=0.2, batch_size=64, epochs=10):
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.epochs = epochs
        self.actor = ActorNetwork(actor_lr, input_dims, fc1_dims, fc2_dims, n_actions)
        self.critic = CriticNetwork(critic_lr, input_dims, fc1_dims, fc2_dims)
        self.memory = {'states': [], 'actions': [], 'log_probs': [], 'rewards': [], 'values': [], 'dones': []}

    def choose_action(self, observation):
        state = T.tensor([observation], dtype=T.float).to(self.actor.device)
        probabilities = self.actor.forward(state)
        dist = T.distributions.Categorical(probabilities)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        value = self.critic.forward(state)
        return action.item(), log_prob.item(), value.item()

    def store_transition(self, state, action, log_prob, reward, value, done):
        self.memory['states'].append(state)
        self.memory['actions'].append(action)
        self.memory['log_probs'].append(log_prob)
        self.memory['rewards'].append(reward)
        self.memory['values'].append(value)
        self.memory['dones'].append(done)

    def clear_memory(self):
        self.memory = {'states': [], 'actions': [], 'log_probs': [], 'rewards': [], 'values': [], 'dones': []}

    def learn(self):
        states = T.tensor(self.memory['states'], dtype=T.float).to(self.actor.device)
        actions = T.tensor(self.memory['actions']).to(self.actor.device)
        log_probs = T.tensor(self.memory['log_probs']).to(self.actor.device)
        rewards = np.array(self.memory['rewards'])
        values = T.tensor(self.memory['values'], dtype=T.float).to(self.actor.device)
        dones = np.array(self.memory['dones'])

        # Compute returns and advantages
        returns = []
        discounted_reward = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                discounted_reward = 0
            discounted_reward = reward + self.gamma * discounted_reward
            returns.insert(0, discounted_reward)
        returns = T.tensor(returns, dtype=T.float).to(self.actor.device)

        advantages = returns - values

        # Optimize policy and value networks
        for _ in range(self.epochs):
            indices = np.arange(len(states))
            np.random.shuffle(indices)
            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                minibatch_indices = indices[start:end]

                minibatch_states = states[minibatch_indices]
                minibatch_actions = actions[minibatch_indices]
                minibatch_log_probs = log_probs[minibatch_indices]
                minibatch_advantages = advantages[minibatch_indices].detach()
                minibatch_returns = returns[minibatch_indices]

                # Compute new log_probs
                new_probs = self.actor.forward(minibatch_states)
                dist = T.distributions.Categorical(new_probs)
                new_log_probs = dist.log_prob(minibatch_actions)

                # Compute the ratio
                ratio = T.exp(new_log_probs - minibatch_log_probs)

                # Clipped objective
                surr1 = ratio * minibatch_advantages
                surr2 = T.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * minibatch_advantages
                actor_loss = -T.min(surr1, surr2).mean()

                # Critic loss
                values = self.critic.forward(minibatch_states)
                critic_loss = (minibatch_returns - values).pow(2).mean()

                # Total loss
                loss = actor_loss + 0.5 * critic_loss
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        self.clear_memory()
