import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam # type: ignore
import tensorflow_probability as tfp
from networks import ActorCriticNetwork
import torch

class Agent:
    def __init__(self, alpha=0.0003, gamma=0.99, n_actions=2):
        self.gamma = gamma
        self.n_actions = n_actions
        self.action = None
        self.action_space = [i for i in range(self.n_actions)]

        self.actor_critic = ActorCriticNetwork(n_actions=n_actions)

        self.actor_critic.compile(optimizer=Adam(learning_rate=alpha))
    
    def choose_action(self, observation, valid_actions=None):
        
        observation = np.clip(observation, -1, 1)
        state = tf.convert_to_tensor([observation], dtype=tf.float32)

        _, probs = self.actor_critic(state)

        # Normalize probabilities
        probs = tf.nn.softmax(probs).numpy()[0]

        if valid_actions is not None:
            # Mask invalid actions
            mask = np.zeros_like(probs)
            mask[valid_actions] = 1
            probs = probs * mask

            if np.sum(probs) == 0:
                # If all probabilities are zero, fallback to uniform distribution
                probs[valid_actions] = 1 / len(valid_actions)
            else:
                # Renormalize probabilities
                probs /= np.sum(probs)

        # Create categorical distribution and sample an action
        action_probabilities = tfp.distributions.Categorical(probs=probs)
        action = action_probabilities.sample()

        # Store and return action
        action = tf.clip_by_value(action, 0, self.n_actions - 1).numpy()
        self.action = action
        return self.action


    def save_models(self, file_path="actor_critic.weights.h5"):
        if not file_path.endswith(".weights.h5"):
            file_path += ".weights.h5"
        print(f"... saving models to {file_path} ...")
        self.actor_critic.save_weights(file_path)

    def load_models(self, file_path="actor_critic.weights.h5"):
        if not file_path.endswith(".weights.h5"):
            file_path += ".weights.h5"
        print(f"... loading models from {file_path} ...")
        self.actor_critic.load_weights(file_path)


        
    def learn(self, state, reward, state_, done):
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        state_ = tf.convert_to_tensor([state_], dtype=tf.float32)
        reward = tf.convert_to_tensor(reward, dtype=tf.float32)

        with tf.GradientTape(persistent=True) as tape:
            state_value, probs = self.actor_critic(state)
            state_value_, _ = self.actor_critic(state_)
            state_value = tf.squeeze(state_value)
            state_value_ = tf.squeeze(state_value_)

            action_probs = tfp.distributions.Categorical(probs=probs)

            # Validate action range
            if self.action >= self.n_actions or self.action < 0:
                print(f"Invalid action: {self.action}. Skipping update.")
                return

            log_prob = action_probs.log_prob(self.action)

            delta = reward + self.gamma * state_value_ * (1 - int(done)) - state_value
            actor_loss = -log_prob * delta
            critic_loss = delta**2
            total_loss = actor_loss + critic_loss

        gradient = tape.gradient(total_loss, self.actor_critic.trainable_variables)
        clipped_gradient = [tf.clip_by_value(g, -1.0, 1.0) for g in gradient]
        self.actor_critic.optimizer.apply_gradients(zip(
            clipped_gradient, self.actor_critic.trainable_variables))

