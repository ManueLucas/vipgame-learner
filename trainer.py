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

def train(path_to_vip_weights, path_to_attacker_weights, path_to_defender_weights, grid_file_path, n_games, agent_to_train, baseline_epsilon=0.25, randomize_spawn_points=False):
    agents = {
        "2": 1, # vip
        "3": 1, # defender
        "4": 1, # attacker
    }

    grid_map = np.loadtxt(grid_file_path, delimiter=",")

    if(randomize_spawn_points):
        grid_map[grid_map > 1] = 0
        grid_map = place_agents(grid_map, agents)

    env = environment.VipGame(grid_map=grid_map)
    input_dims = [grid_map.size]

    eps_dec = (1.0 - baseline_epsilon) / (n_games * env.max_timesteps)
    print(f"eps_dec: {eps_dec}")
    vip_agent = Agent(gamma=0.99, epsilon=1.0, lr=0.003, input_dims=[grid_map.size], batch_size=64, n_actions=4, eps_dec=eps_dec, eps_end=baseline_epsilon)
    attacker_agent = Agent(gamma=0.99, epsilon=1.0, lr=0.003, input_dims=[grid_map.size], batch_size=64, n_actions=8, eps_dec=eps_dec, eps_end=baseline_epsilon)
    defender_agent = Agent(gamma=0.99, epsilon=1.0, lr=0.003, input_dims=[grid_map.size], batch_size=64, n_actions=8, eps_dec=eps_dec, eps_end=baseline_epsilon) 
    
    if os.path.exists(path_to_vip_weights):
        vip_agent.load_weights(path_to_vip_weights)

    if os.path.exists(path_to_attacker_weights):
        attacker_agent.load_weights(path_to_attacker_weights)

    if os.path.exists(path_to_defender_weights):
        defender_agent.load_weights(path_to_defender_weights)
        
    print(env.number_of_attackers)
    print(env.number_of_defenders)
    print(env.number_of_vips)
    scores = {'vip': [], 'attacker': [], 'defender': []}
    eps_history = {'vip': [], 'attacker': [], 'defender': []}


    for i in range(n_games):
        truncated = False
        terminated = False
        observation = env.reset().flatten()
        score = {'vip': 0, 'attacker': 0, 'defender': 0}

        print(f"observation type: {observation.dtype}")
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

            for k in range(env.number_of_attackers):
                if env.attacker_positions[k] == env.dead_cell:
                    attacker_actions.append(-1)
                elif agent_to_train == "attacker":
                    attacker_actions.append(attacker_agent.choose_action(individual_state(observation, env.attacker_positions[k], env.grid_width)))
                elif os.path.exists('attacker_agent_weights.pth'):
                    attacker_actions.append(attacker_agent.choose_action(individual_state(observation, env.attacker_positions[k], env.grid_width)))
                else:
                    attacker_actions.append(np.random.randint(0, attacker_agent.n_actions))

            for k in range(env.number_of_defenders):
                if env.defender_positions[k] == env.dead_cell:
                    defender_actions.append(-1)
                elif agent_to_train == "defender":
                    defender_actions.append(defender_agent.choose_action(individual_state(observation, env.defender_positions[k], env.grid_width)))
                elif os.path.exists('defender_agent_weights.pth'):
                    defender_actions.append(defender_agent.choose_action(individual_state(observation, env.defender_positions[k], env.grid_width)))
                else:
                    defender_actions.append(np.random.randint(0, defender_agent.n_actions))

            for k in range(env.number_of_vips):
                if env.vip_positions[k] == env.dead_cell:
                    vip_actions.append(-1)
                elif agent_to_train == "vip":
                    vip_actions.append(vip_agent.choose_action(individual_state(observation, env.vip_positions[k], env.grid_width)))
                elif os.path.exists('vip_agent_weights.pth'):
                    vip_actions.append(vip_agent.choose_action(individual_state(observation, env.vip_positions[k], env.grid_width)))
                else:
                    vip_actions.append(np.random.randint(0, vip_agent.n_actions))

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
            
            if agent_to_train == "attacker":
                for k in range(env.number_of_attackers):
                    attacker_position = env.attacker_positions[k]
                    if env.attacker_positions[k] != env.dead_cell:
                        attacker_agent.store_transition(individual_state(observation, attacker_position, env.grid_width), actions[0][k], attacker_reward[k], individual_state(observation_, attacker_position, env.grid_width), truncated, attacker_position)
                attacker_agent.learn()

            elif agent_to_train == "defender":
                for k in range(env.number_of_defenders):
                    defender_position = env.defender_positions[k]
                    if env.defender_positions[k] != env.dead_cell:
                        defender_agent.store_transition(individual_state(observation, defender_position, env.grid_width), actions[1][k], defender_reward[k], individual_state(observation_, defender_position, env.grid_width), truncated, defender_position)
                defender_agent.learn()

            elif agent_to_train == "vip":
                for k in range(env.number_of_vips):
                    vip_position = env.vip_positions[k]
                    if env.vip_positions[k] != env.dead_cell:
                        vip_agent.store_transition(individual_state(observation, vip_position, env.grid_width), actions[2][k], vip_reward[k], individual_state(observation_, vip_position, env.grid_width), truncated, vip_position)
                vip_agent.learn()
            
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

        if (i + 1) % 100 == 0:
            if agent_to_train == 'vip':
                vip_agent.save_weights('vip_agent_weights.pth')
            elif agent_to_train == 'attacker':
                attacker_agent.save_weights('attacker_agent_weights.pth')
            elif agent_to_train == 'defender':
                defender_agent.save_weights('defender_agent_weights.pth')
        
    #save the weights of the agent
    if agent_to_train == 'vip':
        vip_agent.save_weights('vip_agent_weights.pth')
    elif agent_to_train == 'attacker':
        attacker_agent.save_weights('attacker_agent_weights.pth')
    elif agent_to_train == 'defender':
        defender_agent.save_weights('defender_agent_weights.pth')

    # Plots n dat
    x = [i + 1 for i in range(n_games)]
    filename = f'{agent_to_train}_training.png'
    plot_learning_curve(x, scores[agent_to_train], eps_history[agent_to_train], filename)
    
def trial(path_to_vip_weights, path_to_attacker_weights, path_to_defender_weights, grid_file_path, epsilon=0.1):
    grid_map = np.loadtxt(grid_file_path, delimiter=",")
    env = environment.VipGame(grid_map=grid_map)
    
    vip_agent = Agent(gamma=0.99, epsilon=epsilon, lr=0.003, input_dims=[grid_map.size], batch_size=64, n_actions=4)
    attacker_agent = Agent(gamma=0.99, epsilon=epsilon, lr=0.003, input_dims=[grid_map.size], batch_size=64, n_actions=8)
    defender_agent = Agent(gamma=0.99, epsilon=epsilon, lr=0.003, input_dims=[grid_map.size], batch_size=64, n_actions=8) 
    
    if(os.path.exists('vip_agent_weights.pth')):
        vip_agent.load_weights('vip_agent_weights.pth')
    if(os.path.exists('attacker_agent_weights.pth')):
        attacker_agent.load_weights('attacker_agent_weights.pth')
    if(os.path.exists('defender_agent_weights.pth')):  
        defender_agent.load_weights('defender_agent_weights.pth')

    truncated = False
    terminated = False
    observation = env.reset().flatten()

    while not (truncated or terminated):
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
                attacker_agent_action = attacker_agent.choose_action(attacker_state)
                attacker_actions.append(attacker_agent_action)
        for j in range(env.number_of_defenders):
            if env.defender_positions[j] == env.dead_cell:
                defender_actions.append(-1)

            else:   
                defender_state = individual_state(observation, env.defender_positions[j], env.grid_width)
                defender_agent_action = defender_agent.choose_action(defender_state)
                defender_actions.append(defender_agent_action)

        for j in range(env.number_of_vips):
            if env.vip_positions[j] == env.dead_cell:
                vip_actions.append(-1)

            else:
                vip_state = individual_state(observation, env.vip_positions[j], env.grid_width)
                vip_agent_action = vip_agent.choose_action(vip_state)
                vip_actions.append(vip_agent_action)

        actions = (attacker_actions, defender_actions, vip_actions)

        fully_visible_state, (defenderside_vision, attackerside_vision), \
        (defender_reward, attacker_reward, vip_reward), \
        (defender_positions, attacker_positions, vip_positions), truncated, terminate = env.step(actions)

        observation_ = fully_visible_state.flatten()
        env.render(fully_visible_state)
        time.sleep(0.05)  # Wait for 100ms

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train or run a trial for the VIP game.')
    parser.add_argument('mode', choices=['train', 'trial'], help='Mode to run: train or trial')
    parser.add_argument('--epsilon', type=float, default=0.1, help='Epsilon value for the trial, or minimum epsilon value for training')
    parser.add_argument('--map', type=str, default='map_presets/grid.csv', help='Path to the map file')
    parser.add_argument('--episodes', type=int, default=10, help='N many episodes to train the agents')
    parser.add_argument('--random', type=bool, default=False, help='randomizes spawn points if set to true')
    parser.add_argument('--agent', type=str, help='specify which agent to train (attacker, defender, vip)')
    parser.add_argument('--cycles', type=int, default=1, help='Number of training cycles to run for all agents')
    args = parser.parse_args()
    
    if args.epsilon < 0 or args.epsilon > 1:
        raise ValueError('Epsilon must be between 0 and 1')


    if args.mode == 'train':
        for cycle in range(args.cycles):
            print(f"TRAINING VIP (Cycle: {cycle}/{args.cycles})")
            train('vip_agent_weights.pth', 'attacker_agent_weights.pth', 'defender_agent_weights.pth', args.map, args.episodes, agent_to_train="vip", baseline_epsilon=args.epsilon, randomize_spawn_points=args.random)
            print(f"TRAINING DEFENDER (Cycle: {cycle}/{args.cycles})")
            train('vip_agent_weights.pth', 'attacker_agent_weights.pth', 'defender_agent_weights.pth', args.map, args.episodes, agent_to_train="defender", baseline_epsilon=args.epsilon, randomize_spawn_points=args.random)
            print(f"TRAINING ATTACKER (Cycle: {cycle}/{args.cycles})")
            train('vip_agent_weights.pth', 'attacker_agent_weights.pth', 'defender_agent_weights.pth', args.map, args.episodes, agent_to_train="attacker", baseline_epsilon=args.epsilon, randomize_spawn_points=args.random)
        print("Training Complete")
    elif args.mode == 'trial':
        trial('vip_agent_weights.pth', 'attacker_agent_weights.pth', 'defender_agent_weights.pth', args.map, epsilon=args.epsilon)