import gymnasium
from simple_dqn_torch import Agent
import numpy as np
import environment
import matplotlib.pyplot as plt
from main import individual_state


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

if __name__ == '__main__':
    grid_map = np.loadtxt("map_presets/grid.csv", delimiter=",")
    env = environment.VipGame(grid_map=grid_map)
    input_dims = [grid_map.size]

    vip_agent = Agent(gamma=0.99, epsilon=1.0, lr=0.003, input_dims=[grid_map.size], batch_size=64, n_actions=4)
    attacker_agent = Agent(gamma=0.99, epsilon=1.0, lr=0.003, input_dims=[grid_map.size], batch_size=64, n_actions=8)
    defender_agent = Agent(gamma=0.99, epsilon=1.0, lr=0.003, input_dims=[grid_map.size], batch_size=64, n_actions=8)
    
    scores = {'vip': [], 'attacker': [], 'defender': []}
    eps_history = {'vip': [], 'attacker': [], 'defender': []}

    n_games = 500

    for i in range(n_games):
        done = False
        observation = env.reset().flatten()
        score = {'vip': 0, 'attacker': 0, 'defender': 0}

        print(f"observation type: {observation.dtype}")

        while not done:
            # attacker_actions = [attacker_agent.choose_action(observation) for _ in range(1)]
            # attacker_actions = [attacker_agent.choose_action(individual_state(observation, env.attacker_positions[i], env.grid_width)) for i in range(1)]
            # defender_actions = [np.random.randint(0, defender_agent.n_actions) for _ in range(1)]
            # defender_actions = [defender_agent.choose_action(individual_state(observation, env.defender_positions[i], env.grid_width)) for i in range(1)]
            # vip_actions = [np.random.randint(0, vip_agent.n_actions) for _ in range(1)]
            # vip_actions = [vip_agent.choose_action(individual_state(observation, env.vip_positions[i], env.grid_width)) for i in range(1)]
            
            attacker_actions = []
            defender_actions = []
            vip_actions = []

            for j in range(1):
                if env.attacker_positions[j] == env.dead_cell:
                    attacker_actions.append(-1)
                    
                else:
                    attacker_state = individual_state(observation, env.attacker_positions[j], env.grid_width)
                    attacker_agent_action = attacker_agent.choose_action(attacker_state)
                    attacker_actions.append(attacker_agent_action)

                if env.defender_positions[j] == env.dead_cell:
                    defender_actions.append(-1)

                else:   
                    defender_state = individual_state(observation, env.defender_positions[j], env.grid_width)
                    defender_agent_action = defender_agent.choose_action(defender_state)
                    defender_actions.append(defender_agent_action)

                if env.vip_positions[j] == env.dead_cell:
                    vip_actions.append(-1)

                else:
                    vip_state = individual_state(observation, env.vip_positions[j], env.grid_width)
                    vip_agent_action = vip_agent.choose_action(vip_state)
                    vip_actions.append(vip_agent_action)

            actions = (attacker_actions, defender_actions, vip_actions)

            fully_visible_state, (defenderside_vision, attackerside_vision), \
            (defender_reward, attacker_reward, vip_reward), \
            (defender_positions, attacker_positions, vip_positions), done = env.step(actions)

            observation_ = fully_visible_state.flatten()

            score['attacker'] += sum(attacker_reward)
            score['defender'] += sum(defender_reward)
            score['vip'] += sum(vip_reward)

            vip_agent.store_transition(observation, actions[2][0], sum(vip_reward), observation_, done)
            vip_agent.learn()

            attacker_agent.store_transition(observation, actions[0][0], sum(attacker_reward), observation_, done)
            attacker_agent.learn()

            defender_agent.store_transition(observation, actions[1][0], sum(defender_reward), observation_, done)
            defender_agent.learn()

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

    # Plots n dat
    for agent_name in ['vip', 'attacker', 'defender']:
        x = [i + 1 for i in range(n_games)]
        filename = f'{agent_name}_training.png'
        plot_learning_curve(x, scores[agent_name], eps_history[agent_name], filename)
