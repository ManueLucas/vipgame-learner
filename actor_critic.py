import numpy as np
# from tensorflow.keras.optimizers import Adam # type: ignore
from networks import ActorCriticNetwork
import torch
import torch.nn as nn
import torch.optim as optim

class Agent:
    def __init__(self, alpha=0.0003, gamma=0.99, n_actions=2):
        self.gamma = gamma
        self.n_actions = n_actions
        self.action = None
        self.action_space = [i for i in range(self.n_actions)]

        self.actor_critic = ActorCriticNetwork(n_actions=n_actions)

        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=alpha)
    
    def choose_action(self, observation, valid_actions=None):
        
        observation = np.clip(observation, -1, 1)
        state = torch.tensor(np.array(observation), dtype=torch.float32).unsqueeze(0)

        _, probs = self.actor_critic(state)

        # Normalize probabilities
        probs = torch.softmax(probs, dim=-1).detach().numpy()[0]

        if valid_actions is not None:
            # Mask invalid actions
            mask = np.zeros_like(probs)
            mask[valid_actions] = 1
            probs = probs * mask

            if np.sum(probs) == 0:
                # If all probabilities are zero, fallback to uniform distribution
                print(f"Warning: All probabilities zero, fallback to uniform.")
                probs[valid_actions] = 1 / len(valid_actions)
            else:
                # Renormalize probabilities
                probs /= np.sum(probs)

        # Create categorical distribution and sample an action
        try:
            action_distribution = torch.distributions.Categorical(torch.tensor(probs))
            action = action_distribution.sample().item()
        except Exception as e:
            print(f"Error in action sampling: {e}, probs={probs}")
            action = np.random.choice(valid_actions) if valid_actions else 0

        # Store and return action
        self.action = action
        return self.action

    def save_models(self, file_path="actor_critic.pth"):
        print(f"... saving models to {file_path} ...")
        torch.save(self.actor_critic.state_dict(), file_path)

    def load_models(self, file_path="actor_critic.pth"):
        print(f"... loading models from {file_path} ...")
        self.actor_critic.load_state_dict(torch.load(file_path))
        
    def learn(self, state, reward, state_, done):
        state = torch.tensor(np.array(state), dtype=torch.float32)
        state_ = torch.tensor(np.array(state_), dtype=torch.float32)
        reward = torch.tensor(reward, dtype=torch.float32)

        if self.action is None:
            print("Error: Action not set, falling back to random.")
            self.action = np.random.randint(0, self.n_actions)

        state_value, probs = self.actor_critic(state)
        state_value_, _ = self.actor_critic(state_)

        state_value = state_value.squeeze()
        state_value_ = state_value_.squeeze()

        action_probs = torch.distributions.Categorical(probs=torch.softmax(probs, dim=-1))
        log_prob = action_probs.log_prob(torch.tensor(self.action, dtype=torch.int64))

        delta = reward + self.gamma * state_value_ * (1 - int(done)) - state_value
        actor_loss = -log_prob * delta
        critic_loss = delta**2
        total_loss = actor_loss + critic_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

