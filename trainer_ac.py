import os
import numpy as np
import argparse
from actor_critic import Agent
from environment import VipGame  # Assuming you have this environment implemented
from utils import plot_learning_curve
import time


def train_actor_critic(
    path_to_weights,
    grid_file_path,
    n_games,
    agent_to_train,
    randomize_spawn_points=False
):
    # Load grid map
    grid_map = np.loadtxt(grid_file_path, delimiter=",")
    if randomize_spawn_points:
        grid_map[grid_map > 1] = 0

    # Initialize environment and agent
    env = VipGame(grid_map=grid_map)
    input_dims = [grid_map.size]
    agent = Agent(alpha=0.0001, gamma=0.99, n_actions=8)

    # Load pre-trained weights if available
    if os.path.exists(path_to_weights):
        agent.load_models(file_path=f"{agent_to_train}_weights.weights.h5")

    scores = []
    for i in range(n_games):
        observation = env.reset().flatten()
        score = 0
        done = False  # To track if the episode should stop

        while not done:
            attacker_actions = [
                agent.choose_action(observation, valid_actions=range(env.attacker_defender_action_space))
                for _ in range(env.number_of_attackers)
            ]
            defender_actions = [
                agent.choose_action(observation, valid_actions=range(env.attacker_defender_action_space))
                for _ in range(env.number_of_defenders)
            ]
            vip_actions = [
                agent.choose_action(observation, valid_actions=range(env.vip_action_space))
                for _ in range(env.number_of_vips)
            ]

            actions = (attacker_actions, defender_actions, vip_actions)
            observation_, visions_tuple, rewards_tuple, positions_tuple, truncated, terminated = env.step(actions)

            observation_ = observation_.flatten()

            reward = sum(map(sum, rewards_tuple))
            score += reward

            # Learn from the transition
            agent.learn(observation, reward, observation_, done)

            # Update observation
            observation = observation_

            # Check if the episode should end
            done = terminated or truncated

        # Log episode stats
        scores.append(score)
        avg_score = np.mean(scores[-100:])
        print(f"Episode {i + 1}/{n_games}, Score: {score}, Avg Score (last 100): {avg_score:.2f}")

        # Save weights periodically
        if (i + 1) % 10 == 0:
            agent.save_models(file_path=f"{agent_to_train}_weights.weights.h5")

    # Save weights after training
    agent.save_models(file_path=f"{agent_to_train}_weights.weights.h5")

    # Plot learning curve
    x = [i + 1 for i in range(len(scores))]
    plot_learning_curve(x, scores, f"{agent_to_train}_training.png")

    return scores


def trial_actor_critic(
    path_to_weights,
    grid_file_path
):
    grid_map = np.loadtxt(grid_file_path, delimiter=",")
    env = VipGame(grid_map=grid_map)

    # Initialize the agent with pre-trained weights
    agent = Agent(alpha=0.0001, gamma=0.99, n_actions=8)
    if os.path.exists(path_to_weights):
        agent.load_models(file_path=f"actor_critic_weights.weights.h5")

    truncated = False
    terminated = False
    observation = env.reset().flatten()

    while not (truncated or terminated):
        attacker_actions = [
            agent.choose_action(observation, valid_actions=range(env.attacker_defender_action_space))
            for _ in range(env.number_of_attackers)
        ]
        defender_actions = [
            agent.choose_action(observation, valid_actions=range(env.attacker_defender_action_space))
            for _ in range(env.number_of_defenders)
        ]
        vip_actions = [
            agent.choose_action(observation, valid_actions=range(env.vip_action_space))
            for _ in range(env.number_of_vips)
        ]

        actions = (attacker_actions, defender_actions, vip_actions)
        observation_, visions_tuple, rewards_tuple, positions_tuple, truncated, terminated = env.step(actions)

        observation = observation_.flatten()

        # Render the environment for visualization
        env.render(env.grid)
        time.sleep(0.1)  # Wait for 100ms to slow down the visualization


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or trial an Actor-Critic agent for the VIP game.")
    parser.add_argument("mode", choices=["train", "trial"], help="Mode to run: train or trial")
    parser.add_argument("--map", type=str, default="map_presets/grid.csv", help="Path to the map file")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to train")
    parser.add_argument("--random", action="store_true", help="Randomizes spawn points if set to true")
    parser.add_argument("--agent", type=str, default="actor_critic", help="Specify the agent to train")
    parser.add_argument("--epsilon", type=float, default=0.1, help="Epsilon value for trials")
    args = parser.parse_args()

    if args.mode == "train":
        scores = train_actor_critic(
            path_to_weights=f"{args.agent}_weights.h5",
            grid_file_path=args.map,
            n_games=args.episodes,
            agent_to_train=args.agent,
            randomize_spawn_points=args.random
        )
        print("Training complete. Learning curve plotted.")

    elif args.mode == "trial":
        trial_actor_critic(
            path_to_weights=f"{args.agent}_weights.h5",
            grid_file_path=args.map
        )
