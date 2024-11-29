from collections import Counter
import gymnasium
from simple_dqn_torch import Agent
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
    
    
def place_agents(grid_map, agents):
    valid_positions = np.argwhere(grid_map == 0)
    
    np.random.shuffle(valid_positions)
    
    for agent_type, count in agents.items():
        for _ in range(count):
            
            pos = valid_positions[0]
            valid_positions = valid_positions[1:]
            
            grid_map[tuple(pos)] = agent_type

    return grid_map

def train(path_to_weights, grid_file_path, n_games, baseline_epsilon=0.25, randomize_spawn_points=False):
    agents = {
        "2": 1, # vip
        "3": 1, # defender
        "4": 1, # attacker
    }
    attackerVector = np.array([1., 0., 0.])
    vipVector = np.array([0., 1., 0.])

    defenderVector = np.array([0., 0., 1.])

    grid_map = np.loadtxt(grid_file_path, delimiter=",")

    if(randomize_spawn_points):
        grid_map[grid_map > 1] = 0
        grid_map = place_agents(grid_map, agents)

    env = environment.VipGame(grid_map=grid_map)
    input_dims = [grid_map.size]

    eps_dec = 1/(n_games * env.max_timesteps)
    print(f"eps_dec: {eps_dec}")
    agent = Agent(gamma=0.99, epsilon=1.0, lr=0.003, input_dims=[grid_map.size + 3], batch_size=64, n_actions=8, eps_dec=eps_dec, eps_end=baseline_epsilon)
    
    if(os.path.exists(path_to_weights)):
        agent.load_weights(path_to_weights)
        
    print(env.number_of_attackers)
    print(env.number_of_defenders)
    print(env.number_of_vips)
    scores = {'vip': [], 'attacker': [], 'defender': []}
    eps_history = []
    total_steps = 0
    target_steps = n_games*env.max_timesteps

    # Initialize action counters for each agent type
    action_counters = {
        "attacker": Counter(),
        "defender": Counter(),
        "vip": Counter()
    }

    for i in range(n_games):
        truncated = False
        terminated = False
        observation = env.reset().flatten()
        score = {'vip': 0, 'attacker': 0, 'defender': 0}
        episode_steps = 0

        # print(f"observation type: {observation.dtype}")
        # print(f'team vector type: {vipVector.dtype}')
        # env.render(grid_map)

        while True:
            if(truncated or terminated):
                print('we should have terminated by now')
            # attacker_actions = [attacker_agent.choose_action(observation) for _ in range(1)]
            # attacker_actions = [attacker_agent.choose_action(individual_state(observation, env.attacker_positions[i], env.grid_width)) for i in range(1)]
            # defender_actions = [np.random.randint(0, defender_agent.n_actions) for _ in range(1)]
            # defender_actions = [defender_agent.choose_action(individual_state(observation, env.defender_positions[i], env.grid_width)) for i in range(1)]
            # vip_actions = [np.random.randint(0, vip_agent.n_actions) for _ in range(1)]
            # vip_actions = [vip_agent.choose_action(individual_state(observation, env.vip_positions[i], env.grid_width)) for i in range(1)]
            
            attacker_actions = []
            defender_actions = []
            vip_actions = []
            

            for j in range(env.number_of_attackers):
                if env.attacker_positions[j] == env.dead_cell:
                    attacker_actions.append(-1)
                    
                else:
                    attacker_state = individual_state(observation, env.attacker_positions[j], env.grid_width)
                    attacker_agent_action = agent.choose_action(attacker_state, attackerVector)
                    attacker_actions.append(attacker_agent_action)
                    action_counters["attacker"][attacker_agent_action] += 1
           # print(f"attacker_actions: {attacker_actions}")
            for j in range(env.number_of_defenders):
                if env.defender_positions[j] == env.dead_cell:
                    defender_actions.append(-1)

                else:   
                    defender_state = individual_state(observation, env.defender_positions[j], env.grid_width)
                    defender_agent_action = agent.choose_action(defender_state, defenderVector)
                    defender_actions.append(defender_agent_action)
                    action_counters["defender"][defender_agent_action] += 1

            for j in range(env.number_of_vips):
                if env.vip_positions[j] == env.dead_cell:
                    vip_actions.append(-1)

                else:
                    vip_state = individual_state(observation, env.vip_positions[j], env.grid_width)
                    vip_agent_action = agent.choose_action(vip_state, vipVector)
                    vip_actions.append(vip_agent_action)
                    action_counters["vip"][vip_agent_action] += 1

            actions = (attacker_actions, defender_actions, vip_actions)

            fully_visible_state, (defenderside_vision, attackerside_vision), \
            (defender_reward, attacker_reward, vip_reward), \
            (defender_positions, attacker_positions, vip_positions), truncated, terminated = env.step(actions)
            
            # if(attacker_reward[0] != 0 or defender_reward[0] != 0 or vip_reward[0] != 0):
            #     print(f'attacker_reward: {attacker_reward}')
            #     print(f'defender_reward: {defender_reward}')
            #     print(f'vip_reward: {vip_reward}')
            observation_ = fully_visible_state.flatten()

            score['attacker'] += np.mean(attacker_reward)
            score['defender'] += np.mean(defender_reward)
            score['vip'] += np.mean(vip_reward)
            
            for k in range(env.number_of_attackers):
                attacker_position = env.attacker_positions[k]
                if env.attacker_positions[k] != env.dead_cell:
                    agent.store_transition(individual_state(observation, attacker_position, env.grid_width), actions[0][k], attacker_reward[k], individual_state(observation_, attacker_position, env.grid_width), truncated, attacker_position, attackerVector)
            
            for k in range(env.number_of_defenders):
                defender_position = env.defender_positions[k]
                if env.defender_positions[k] != env.dead_cell:
                    agent.store_transition(individual_state(observation, defender_position, env.grid_width), actions[1][k], defender_reward[k], individual_state(observation_, defender_position, env.grid_width), truncated, defender_position, defenderVector)
            
            for k in range(env.number_of_vips):
                vip_position = env.vip_positions[k]
                if env.vip_positions[k] != env.dead_cell:
                    agent.store_transition(individual_state(observation, vip_position, env.grid_width), actions[2][k], vip_reward[k], individual_state(observation_, vip_position, env.grid_width), truncated, vip_position, vipVector)
            
            agent.learn()
            episode_steps += 1
            total_steps += 1
            
            if truncated or terminated:
                break
            observation = observation_
        
        for agent_type in ['vip', 'attacker', 'defender']:
            scores[agent_type].append(score[agent_type])
        eps_history.append(agent.epsilon)

        # Dynamically adjust epsilon decay based on steps
        agent.epsilon = max(agent.epsilon - (env.max_timesteps - env.timesteps_elapsed)*(eps_dec), baseline_epsilon)

        # print(f"Episode {i}")
        # print(f"  VIP: Score {score['vip']:.2f}, Avg Score {np.mean(scores['vip'][-100:]):.2f}, Epsilon {agent.epsilon:.2f}")
        # print(f"  Attacker: Score {score['attacker']:.2f}, Avg Score {np.mean(scores['attacker'][-100:]):.2f}, Epsilon {agent.epsilon:.2f}")
        # print(f"  Defender: Score {score['defender']:.2f}, Avg Score {np.mean(scores['defender'][-100:]):.2f}, Epsilon {agent.epsilon:.2f}")

        if (i + 1) % 100 == 0:
            agent.save_weights('agent_weights.pth')
        
    #save the weights of the agent
    agent.save_weights('agent_weights.pth')

    # Print action counts at the end of training
    print("\nAction Counts:")
    for agent_type, counter in action_counters.items():
        print(f"{agent_type.capitalize()} actions:")
        for action, count in sorted(counter.items()):
            print(f"  Action {list(environment.ACTIONS.keys())[action]}: {count} times")

    # Plots n dat
    for i in ['vip', 'attacker', 'defender']:
        x = [i + 1 for i in range(n_games)]
        plot_learning_curve(x, np.mean(list(scores.values()), axis=0), eps_history, 'training.png')

    
def trial(path_to_weights, grid_file_path, epsilon=0.1):
    grid_map = np.loadtxt(grid_file_path, delimiter=",")
    env = environment.VipGame(grid_map=grid_map)

    attackerVector = np.array([1., 0., 0.])
    vipVector = np.array([0., 1., 0.])
    defenderVector = np.array([0., 0., 1.])

    # Single agent for all roles
    agent = Agent(gamma=0.99, epsilon=epsilon, lr=0.003, input_dims=[grid_map.size + 3], batch_size=64, n_actions=8)

    # Load weights if they exist
    if os.path.exists(path_to_weights):
        agent.load_weights(path_to_weights)

    truncated = False
    terminated = False
    observation = env.reset().flatten()

    while not (truncated or terminated):
        attacker_actions, defender_actions, vip_actions = [], [], []

        # Generate actions for each agent
        for j in range(env.number_of_attackers):
            if env.attacker_positions[j] == env.dead_cell:
                attacker_actions.append(-1)
            else:
                attacker_state = individual_state(observation, env.attacker_positions[j], env.grid_width)
                attacker_action = agent.choose_action(attacker_state, attackerVector)
                print(f'{list(environment.ACTIONS.keys())[attacker_action]}')
                attacker_actions.append(attacker_action)

        for j in range(env.number_of_defenders):
            if env.defender_positions[j] == env.dead_cell:
                defender_actions.append(-1)
            else:
                defender_state = individual_state(observation, env.defender_positions[j], env.grid_width)
                defender_action = agent.choose_action(defender_state, defenderVector)
                print(f'{list(environment.ACTIONS.keys())[defender_action]}')
                defender_actions.append(defender_action)

        for j in range(env.number_of_vips):
            if env.vip_positions[j] == env.dead_cell:
                vip_actions.append(-1)
            else:
                vip_state = individual_state(observation, env.vip_positions[j], env.grid_width)
                vip_action = agent.choose_action(vip_state, vipVector)
                print(f'{list(environment.ACTIONS.keys())[vip_action]}')
                vip_actions.append(vip_action)


        actions = (attacker_actions, defender_actions, vip_actions)

        # Step the environment
        fully_visible_state, _, _, _, truncated, terminated = env.step(actions)

        # Update observation
        observation = fully_visible_state.flatten()

        # Render environment for visualization
        env.render(fully_visible_state)
        time.sleep(0.05)  # Pause for visualization

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train or run a trial for the VIP game.')
    parser.add_argument('mode', choices=['train', 'trial'], help='Mode to run: train or trial')
    parser.add_argument('--epsilon', type=float, default=0.1, help='Epsilon value for the trial or minimum epsilon value for training')
    parser.add_argument('--map', type=str, default='map_presets/grid.csv', help='Path to the map file')
    parser.add_argument('--episodes', type=int, default=10, help='Number of episodes to train the agent')
    parser.add_argument('--random', type=bool, default=False, help='Randomize spawn points if set to true')
    args = parser.parse_args()

    if args.epsilon < 0 or args.epsilon > 1:
        raise ValueError('Epsilon must be between 0 and 1')

    if args.mode == 'train':
        train('agent_weights.pth', args.map, args.episodes, baseline_epsilon=args.epsilon, randomize_spawn_points=args.random)
    elif args.mode == 'trial':
        trial('agent_weights.pth', args.map, epsilon=args.epsilon)