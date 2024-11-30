import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

# Define the Actor Network
class Actor(nn.Module):
    def __init__(self, input_size, action_size):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return torch.softmax(self.fc2(x), dim=-1)  # Output action probabilities

# Define the Critic Network
class Critic(nn.Module):
    def __init__(self, input_size):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 1)  # Output value estimation

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)  # Output state value

# Actor-Critic agent
class ActorCriticTrainer:
    def __init__(self, actor, critic, critic_optimizer, learning_rate=0.001):
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)
        
    def select_action(self, state):
        state_tensor = torch.FloatTensor(state)
        action_probs = self.actor(state_tensor)
        action = np.random.choice(len(action_probs), p=action_probs.detach().numpy())
        return action

    def train(self, state, action, reward, next_state, done):
        state_tensor = torch.FloatTensor(state)
        next_state_tensor = torch.FloatTensor(next_state)
        action_tensor = torch.LongTensor([action])

        # Calculate the value and advantage
        value = self.critic(state_tensor)
        next_value = self.critic(next_state_tensor) * (1 - int(done))  # if not done, get next value
        target = reward + next_value.detach()  # Target for the critic
        advantage = target - value  # Advantage for the actor

        # Update the critic
        critic_loss = nn.MSELoss()(value, target.detach())
        print(critic_loss)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update the actor
        action_log_probs = torch.log(self.actor(state_tensor)[action_tensor])  # Log probability of taken action
        actor_loss = -action_log_probs * advantage.detach()  # Actor loss based on advantage
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

# Example usage
if __name__ == "__main__":
    # Environment parameters
    state_size = 4  # Example state size (e.g., CartPole state space)
    action_size = 2  # Example action space (e.g., left or right)

    # Create agent
    agent = ActorCriticTrainer(state_size, action_size)

    # Training loop
    for episode in range(1000):  # Run for 1000 episodes
        state = np.random.randn(state_size)  # Initialize state (for example)
        done = False

        while not done:
            action = agent.select_action(state)
            next_state = np.random.rand(state_size)  # Simulate next state
            reward = random.choice([1, 0])  # Simulate reward
            done = np.random.rand() > 0.9  # Randomly end episode

            # Train the agent
            agent.train(state, action, reward, next_state, done)
            state = next_state

