
import gymnasium
from simple_ppo_torch import Agent
import numpy as np
import environment
from main import individual_state
import argparse

def train(path_to_vip_weights, path_to_attacker_weights, path_to_defender_weights, grid_file_path, n_games, baseline_epsilon=0.25, randomize_spawn_points=False):
    agents = {
        "2": 1, # vip
        "3": 1, # defender
        "4": 1, # attacker
    }

    grid_map = np.loadtxt(grid_file_path, delimiter=",")
    env = environment.VipGame(grid_map=grid_map)

    vip_agent = Agent(actor_lr=0.003, critic_lr=0.003, input_dims=grid_map.size, n_actions=4)
    attacker_agent = Agent(actor_lr=0.003, critic_lr=0.003, input_dims=grid_map.size, n_actions=8)
    defender_agent = Agent(actor_lr=0.003, critic_lr=0.003, input_dims=grid_map.size, n_actions=8)

    scores = {'vip': [], 'attacker': [], 'defender': []}

    for i in range(n_games):
        observation = env.reset().flatten()
        score = {'vip': 0, 'attacker': 0, 'defender': 0}
        truncated, terminated = False, False

        while not (truncated or terminated):
            attacker_actions, defender_actions, vip_actions = [], [], []

            for j in range(env.number_of_attackers):
                if env.attacker_positions[j] != env.dead_cell:
                    attacker_state = individual_state(observation, env.attacker_positions[j], env.grid_width)
                    attacker_action, attacker_log_prob, attacker_value = attacker_agent.choose_action(attacker_state)
                    attacker_actions.append(attacker_action)
                    attacker_agent.store_transition(attacker_state, attacker_action, attacker_log_prob, 0, attacker_value, truncated)

            for j in range(env.number_of_defenders):
                if env.defender_positions[j] != env.dead_cell:
                    defender_state = individual_state(observation, env.defender_positions[j], env.grid_width)
                    defender_action, defender_log_prob, defender_value = defender_agent.choose_action(defender_state)
                    defender_actions.append(defender_action)
                    defender_agent.store_transition(defender_state, defender_action, defender_log_prob, 0, defender_value, truncated)

            for j in range(env.number_of_vips):
                if env.vip_positions[j] != env.dead_cell:
                    vip_state = individual_state(observation, env.vip_positions[j], env.grid_width)
                    vip_action, vip_log_prob, vip_value = vip_agent.choose_action(vip_state)
                    vip_actions.append(vip_action)
                    vip_agent.store_transition(vip_state, vip_action, vip_log_prob, 0, vip_value, truncated)

            actions = (attacker_actions, defender_actions, vip_actions)

            # Updated unpacking
            fully_visible_state, (defenderside_vision, attackerside_vision),             (defender_reward, attacker_reward, vip_reward),             (defender_positions, attacker_positions, vip_positions),             truncated, terminated = env.step(actions)

            observation_ = fully_visible_state.flatten()
            score['attacker'] += np.mean(attacker_reward)
            score['defender'] += np.mean(defender_reward)
            score['vip'] += np.mean(vip_reward)
            observation = observation_

        vip_agent.learn()
        attacker_agent.learn()
        defender_agent.learn()

        scores['vip'].append(score['vip'])
        scores['attacker'].append(score['attacker'])
        scores['defender'].append(score['defender'])

        print(f"Episode {i}: VIP {score['vip']:.2f}, Attacker {score['attacker']:.2f}, Defender {score['defender']:.2f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train PPO agents in the VIP game.')
    parser.add_argument('mode', choices=['train'], help='Mode to run: train')
    parser.add_argument('--map', type=str, default='map_presets/grid.csv', help='Path to the map file')
    parser.add_argument('--episodes', type=int, default=10, help='Number of training episodes')
    args = parser.parse_args()

    if args.mode == 'train':
        train('vip_agent_weights.pth', 'attacker_agent_weights.pth', 'defender_agent_weights.pth', args.map, args.episodes)
