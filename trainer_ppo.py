
import gymnasium
from simple_ppo_torch import Agent
import numpy as np
import environment
import time
import matplotlib.pyplot as plt
from main import individual_state
import os
import argparse


def plot_learning_curve(x, scores, epsilons, filename, lines=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, label="1")
    ax2 = fig.add_subplot(111, label="2", frame_on=False)

    ax.plot(x, epsilons, color="C0")
    ax.set_xlabel("Training Steps", color="C0")
    ax.set_ylabel("Epsilon", color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")

    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(scores[max(0, t - 20):(t + 1)])

    ax2.scatter(x, running_avg, color="C1")
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    ax2.set_ylabel('Score', color="C1")
    ax2.yaxis.set_label_position('right')
    ax2.tick_params(axis='y', colors="C1")

    if lines is not None:
        for line in lines:
            plt.axvline(x=line)

    plt.savefig(filename)
    

def train(vip_critic_weights, vip_actor_weights, attacker_critic_weights, attacker_actor_weights, defender_critic_weights, defender_actor_weights, grid_file_path, n_games, baseline_epsilon=0.25):
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
    
    if(os.path.exists(vip_critic_weights) and os.path.exists(vip_actor_weights)):
        vip_agent.load_weights(vip_critic_weights, vip_actor_weights)
    if(os.path.exists(attacker_critic_weights) and os.path.exists(attacker_actor_weights)):
        attacker_agent.load_weights(attacker_critic_weights, attacker_actor_weights)
    if(os.path.exists(defender_critic_weights) and os.path.exists(defender_actor_weights)):
        defender_agent.load_weights(defender_critic_weights, defender_actor_weights)        

    scores = {'vip': [], 'attacker': [], 'defender': []}
    eps_history = {'vip': [], 'attacker': [], 'defender': []}


    for i in range(n_games):
        observation = env.reset().flatten()
        score = {'vip': 0, 'attacker': 0, 'defender': 0}
        truncated, terminated = False, False

        while True:
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
            if truncated or terminated:
                break
            observation = observation_

        scores['vip'].append(score['vip'])
        scores['attacker'].append(score['attacker'])
        scores['defender'].append(score['defender'])

        eps_history['vip'].append(vip_agent.epsilon)
        eps_history['attacker'].append(attacker_agent.epsilon)
        eps_history['defender'].append(defender_agent.epsilon)


        print(f"Episode {i}")
        print(f"  VIP: Score {score['vip']:.2f}, Avg Score {np.mean(scores['vip'][-100:]):.2f}, Epsilon {vip_agent.epsilon:.2f}")
        print(f"  Attacker: Score {score['attacker']:.2f}, Avg Score {np.mean(scores['attacker'][-100:]):.2f}, Epsilon {attacker_agent.epsilon:.2f}")
        print(f"  Defender: Score {score['defender']:.2f}, Avg Score {np.mean(scores['defender'][-100:]):.2f}, Epsilon {defender_agent.epsilon:.2f}")
        
    #save the weights of the agent
    vip_agent.save_weights('vip_agent_weights.pth')
    attacker_agent.save_weights('attacker_agent_weights.pth')
    defender_agent.save_weights('defender_agent_weights.pth')

    # Plots n dat
    for agent_name in ['vip', 'attacker', 'defender']:
        x = [i + 1 for i in range(n_games)]
        filename = f'{agent_name}_training.png'
        plot_learning_curve(x, scores[agent_name], eps_history[agent_name], filename)
    
def trial(path_to_vip_weights, path_to_attacker_weights, path_to_defender_weights, grid_file_path, epsilon=0.1):
    grid_map = np.loadtxt(grid_file_path, delimiter=",")
    env = environment.VipGame(grid_map=grid_map)
    
    vip_agent = Agent(actor_lr=0.003, critic_lr=0.003, input_dims=grid_map.size, n_actions=4)
    attacker_agent = Agent(actor_lr=0.003, critic_lr=0.003, input_dims=grid_map.size, n_actions=8)
    defender_agent = Agent(actor_lr=0.003, critic_lr=0.003, input_dims=grid_map.size, n_actions=8)
    
    if(os.path.exists(vip_critic_weights) and os.path.exists(vip_actor_weights)):
        vip_agent.load_weights(vip_critic_weights, vip_actor_weights)
    if(os.path.exists(attacker_critic_weights) and os.path.exists(attacker_actor_weights)):
        attacker_agent.load_weights(attacker_critic_weights, attacker_actor_weights)
    if(os.path.exists(defender_critic_weights) and os.path.exists(defender_actor_weights)):
        defender_agent.load_weights(defender_critic_weights, defender_actor_weights)        


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
        env.render(fully_visible_state)
        time.sleep(0.05)  # Wait for 100ms

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train or run a trial for the VIP game.')
    parser.add_argument('mode', choices=['train', 'trial'], help='Mode to run: train or trial')
    parser.add_argument('--epsilon', type=float, default=0.1, help='Epsilon value for the trial, or minimum epsilon value for training')
    parser.add_argument('--map', type=str, default='map_presets/grid.csv', help='Path to the map file')
    parser.add_argument('--episodes', type=int, default=10, help='N many episodes to train the agents')
    parser.add_argument('--random', type=bool, default=False, help='randomizes spawn points if set to true')
    args = parser.parse_args()
    
    if args.epsilon < 0 or args.epsilon > 1:
        raise ValueError('Epsilon must be between 0 and 1')
    

    if args.mode == 'train':
        train('vip_agent_critic_weights.pth', 'vip_agent_actor_weights.pth', 'attacker_agent_critic_weights.pth', 'attacker_agent_actor_weights.pth', 'defender_agent_critic_weights.pth', 'defender_agent_actor_weights.pth', args.map, args.episodes, baseline_epsilon=args.epsilon)
    elif args.mode == 'trial':
        trial('vip_agent_critic_weights.pth', 'vip_agent_actor_weights.pth', 'attacker_agent_critic_weights.pth', 'attacker_agent_actor_weights.pth', 'defender_agent_critic_weights.pth', 'defender_agent_actor_weights.pth', args.map, epsilon=args.epsilon) 
