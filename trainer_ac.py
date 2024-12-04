import os
import numpy as np
import argparse
from actor_critic import Agent
from environment import VipGame  # Assuming you have this environment implemented
from utils import plot_learning_curve
from torch.utils.tensorboard import SummaryWriter
import time
import datetime

def place_agents(grid_map, agents):
    valid_positions = np.argwhere(grid_map == 0)
    
    np.random.shuffle(valid_positions)
    
    for agent_type, count in agents.items():
        for _ in range(count):
            
            pos = valid_positions[0]
            valid_positions = valid_positions[1:]
            
            grid_map[tuple(pos)] = agent_type

    return grid_map

def train_actor_critic(path_to_weights,grid_file_path,n_games,agent_to_train,randomize_spawn_points=False):
    agents = {
        "2": 1, # vip
        "3": 1, # defender
        "4": 1, # attacker
    }

    grid_map = np.loadtxt(grid_file_path, delimiter=",")
    agent = Agent(alpha=0.0001, gamma=0.99, n_actions=8)
    

    # Load pre-trained weights if available
    # if os.path.exists(path_to_weights):
    #     agent.load_models(file_path=f"{agent_to_train}_weights.h5")

    scores = []
    for i in range(n_games):
        # Load grid map
        if randomize_spawn_points:
            grid_map[grid_map > 1] = 0
            grid_map = place_agents(grid_map, agents)

        # Initialize environment and agent
        env = VipGame(grid_map=grid_map)
        env.render(env.grid)

        input_dims = [grid_map.size]

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
            agent.save_models(file_path=f"{path_to_weights}/{agent_to_train}_{i+1}_weights.h5")
            # Plot learning curve
            x = [j + 1 for j in range(len(scores))]
            plot_learning_curve(x, scores, f"{path_to_weights}/{agent_to_train}_{i+1}_training.png")

        # Log to TensorBoard
        writer.add_scalar('Reward', reward, i)
        writer.add_scalar('Score', score, i)
        writer.add_scalar('Avg score', avg_score, i)
    writer.close()

    # Save weights after training
    agent.save_models(file_path=f"{path_to_weights}/{agent_to_train}_{i}_weights.h5")

    # Plot learning curve
    x = [i + 1 for i in range(len(scores))]
    plot_learning_curve(x, scores, f"{path_to_weights}/{agent_to_train}_training.png")

    return scores


def trial_actor_critic(path_to_weights, grid_file_path, load_model_num):
    grid_map = np.loadtxt(grid_file_path, delimiter=",")
    env = VipGame(grid_map=grid_map)

    # Initialize the agent with pre-trained weights
    agent = Agent(alpha=0.0001, gamma=0.99, n_actions=8)
    if os.path.exists(path_to_weights):
        agent.load_models(file_path=f"{path_to_weights}/actor_critic_{load_model_num}_weights.h5")

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
    parser.add_argument("--trial_number", type=int, help="Episode number you want to run trial on")
    args = parser.parse_args()

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"{args.agent}_{args.mode}_{current_time}"

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{k}|{v}|" for k, v in vars(args).items()])),
    )
    if args.mode == "train":
        scores = train_actor_critic(
            path_to_weights=f"weights/",
            grid_file_path=args.map,
            n_games=args.episodes,
            agent_to_train=args.agent,
            randomize_spawn_points=args.random
        )
        print("Training complete. Learning curve plotted.")

    elif args.mode == "trial":
        trial_actor_critic(
            path_to_weights=f"weights/",
            grid_file_path=args.map,
            load_model_num=args.trial_number
        )
