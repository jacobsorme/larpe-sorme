import numpy as np
import matplotlib.pyplot as plt
from enum import Enum

city =  np.array([
    [1,0,0,0,0,1],
    [0,0,0,0,0,0],
    [1,0,0,0,0,1]])

walls =  np.array([
    [1,1,1,1,1,1],
    [1,0,0,0,0,1],
    [1,1,1,1,1,1]])

pos0 = np.array([0,0])
pol_station = np.array([1,2])
bank_pos = np.array([[0,0], [0,5], [2,0], [2,5]])

actions = np.array([[0,0], [1,0], [-1,0], [0,1], [0,-1]], dtype=int)
actions_cop = np.array([[1,0], [-1,0], [0,1], [0,-1]], dtype=int)

# All possible positions.
positions = np.array([[row, col] for row in range(3) for col in range(6)])

n_states = city.size * city.size
n_actions = actions.shape[0]

def plot_value_fun():
    epsilon = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]
    epsilon = list(map(lambda x: "Epsilon: "+str(x),epsilon))
    files = [   "sarsa_100000_100_0.01.npy",
                ]
    for f in files:
        plt.plot(np.load(f))
    plt.legend(epsilon)
    plt.title("T = 100")
    plt.ylabel("V[s0]")
    plt.xlabel("Episodes")
    plt.show()


def main():
    #ass()
    #exit()
    K = 100000
    T = 100
    epsilon = 0.01

    # V_hist = np.load(f'sarsa_{K}_{T}_{epsilon}.npy')
    # plt.figure(figsize=(8,3.5))
    # plt.plot(V_hist)
    # plt.title("Q-learning convergence (T=100)")
    # plt.ylabel("V[s0]")
    # plt.xlabel("Episodes")
    # plt.tight_layout()
    # plt.savefig('q_conv')
    # plt.show()
    # plt.exit()

    p, r = calc_prob()
    gamma = 0.8
    epsilon = 0.0000001
    N = 200
    V, policy, V0_hist = value_iteration(p, r, gamma, epsilon, N)

    plt.plot(V0_hist)
    plt.show()
    exit()
    hist, reward = simulate(Q_star, T)
    print('Total reward: ' + str(reward))
    animate(hist)

def create_policy(Q):
    return np.argmax(Q, axis=1) # Return S x 1 vector.

def get_state_grid(pos):
    return int(pos[0] + pos[1] * 3)


def get_state(thief,cop):
    return get_state_grid(thief) + 18*get_state_grid(cop)

def cv(p):
    if p[0] < 0 or p[0] >= 3 or p[1] < 0 or p[1] >= 6:
        return 1
    return 0

def move(pos, delta):
    new = pos + delta
    if new[0] < 0 or new[0] >= mapsize or new[1] < 0 or new[1] >= mapsize:
        return pos
    return new

def calc_prob():
    # p = (cur_state, action, next_state)
    p = np.zeros((n_states, n_actions, n_states))
    r = np.zeros((n_states, n_actions))

    init_s = get_state(pos0, pol_station)

    for p_pos in positions: # player
        for c_pos in positions: # minotaur
            cur_s = get_state(p_pos, c_pos)

            # Caught by cop.
            if np.array_equal(p_pos, c_pos):
                p[cur_s, :, init_s] = 1
                r[cur_s, :] = -50
                continue
            # Rob a bank.
            if city[p_pos[0], p_pos[1]] == 1:
                r[cur_s, :] = 10

            dydx = np.abs(p_pos - c_pos)
            same_y = dydx[0] == 0
            same_x = dydx[1] == 0
            manhattan_d = np.sum(dydx)

            valid_actions = []
            for a in actions_cop:
                new_pos = c_pos + a
                if cv(new_pos):
                    continue
                if same_y:
                    if abs(new_pos[0] - p_pos[0]) > dydx[0]:
                        continue
                elif same_x:
                    if abs(new_pos[1] - p_pos[1]) > dydx[1]:
                        continue
                elif np.sum(np.abs(new_pos - p_pos)) > manhattan_d:
                    continue
                valid_actions.append(a)
            cop_prob = 1 / len(valid_actions)

            for c_act in valid_actions: # cop actions
                c_newpos = c_pos + c_act
                for p_act in range(5): # player actions
                    p_newpos = p_pos + actions[p_act]
                    if not cv(p_newpos):
                        p[cur_s, p_act, get_state(p_newpos, c_newpos)] = cop_prob
                    # if p_pos[0] > c_pos[0]: # Player below
                    #     if p_pos[1] > c_pos[1]: # Player to right
                    #         if a == [1,0] or a == [0,1]: # OK to go right and down
                    #             p[cur_s,p_act,encode_state(p_newpos,c_newpos)] = 1/2
                    #     elif p_pos[1] < c_pos[1]: # Player to left
                    #         if a == [1,0] or a == [0,-1]:# OK to go left and down
                    #             p[cur_s,p_act,encode_state(p_newpos,c_newpos)] = 1/2
    return p, r

def simulate(Q, T, epsilon=0.0):
    if epsilon < 1:
        pi = create_policy(Q)
    # 1 row for every set of s, a and r.
    history = np.zeros((T+1, 4), dtype=int) # +1 for the state s(T+1).

    thief = np.array([0,0])
    police = np.array([3,3])
    history[0, 0] = get_state(thief, police)
    history[0, 3] = get_state_grid(police)
    sum_reward = 0

    # Simulate and store observations.
    for t in range(1, T + 1):
        # Move thief and player.
        if np.random.rand() >= epsilon:
            thief_action = pi[get_state(thief, police)]
        else:
            thief_action = np.random.randint(0,5)

        thief = move(thief, actions[thief_action])

        old_police = np.copy(police)
        while np.array_equal(police, old_police):
            police = move(police, actions[np.random.randint(1,5)])

        reward = 0
        # Reward for being at bank.
        if thief[0] == 1 and thief[1] == 1:
            reward += 1

        # Cost for being caught by police.
        if np.array_equal(police, thief):
            reward -= 10

        # Update history.
        history[t, 0]   = get_state(thief, police)
        history[t-1, 1] = thief_action
        history[t-1, 2] = reward
        history[t, 3]   = get_state_grid(police)
        sum_reward += reward

    # Add last action, only used with SARSA.
    if epsilon < 1:
        history[T, 1] = pi[history[T, 0]]

    return history, sum_reward

def value_iteration(p, r, gamma, epsilon, N):
    n_states  = p.shape[0]
    n_actions = p.shape[1]

    V = np.zeros(n_states)
    Q = np.zeros((n_states, n_actions))
    best_V = np.copy(V)
    V0_hist = np.zeros(N)
    init_s = get_state(pos0, pol_station)

    tol = (1 - gamma) * epsilon / gamma

    # Initialization of the VI
    for s in range(n_states):
        for a in range(n_actions):
            Q[s, a] = r[s, a] + gamma*np.dot(p[s,a], V);
    best_V = np.max(Q, 1);

    # Iterate until convergence
    n = 0
    while np.linalg.norm(V - best_V) >= tol and n < N:
        n += 1
        V = np.copy(best_V)
        # Compute the new best_V
        for s in range(n_states):
            for a in range(n_actions):
                Q[s, a] = r[s, a] + gamma*np.dot(p[s,a], V)
        best_V = np.max(Q, 1)
        # Show error
        print(np.linalg.norm(V - best_V))
        V0_hist[n-1] = best_V[init_s]

    # Compute policy
    policy = np.argmax(Q,1)
    # Return the obtained policy
    return V, policy, V0_hist[:n]


def draw_city():
    # Give a color to each cell
    rows,cols    = city.shape[0], city.shape[1]
    colored_city = [['#ffffff' if city[j,i] == 0 else '#003399' for i in range(cols)] for j in range(rows)]

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows));

    # Remove the axis ticks and add title
    plt.gca().set_xticks([]);
    plt.gca().set_yticks([]);

    # Create a table to colorgrid.get_celld()[(pathP[i])]
    grid = plt.table(cellText=None, cellColours=colored_city, cellLoc='center',
                     loc=(0,0), edges='closed');
    # Modify the hight and width of the cells in the table
    tc = grid.properties()['child_artists']
    for cell in tc:
        cell.set_height(1.0/rows);
        cell.set_width(1.0/cols);

    return grid

def row_col(s):
    s %= 16 # Remove police from encoded state.
    return s // mapsize, s % mapsize

def animate(hist):
    grid = draw_city()
    T = hist.shape[0]

    plt.ion()
    for t in range(T):
        for row in range(mapsize):
            for col in range(mapsize):
                grid.get_celld()[(row, col)].set_facecolor('#ffffff')
                grid.get_celld()[(row, col)].get_text().set_text('')

        # Set bank color.
        grid.get_celld()[(1, 1)].set_facecolor('#3355ee')

        # Animate player
        row, col = row_col(hist[t, 0])
        grid.get_celld()[(row, col)].get_text().set_text('Player')

        # Animate police
        prow, pcol = row_col(hist[t, 3])
        grid.get_celld()[(prow, pcol)].get_text().set_text('Cop')

        if row == prow and col == pcol:
            grid.get_celld()[(row, col)].set_facecolor('#aa3355')

        plt.show()
        plt.pause(0.3)
    plt.ioff()
    plt.pause(1)
    plt.show() # Keep graph alive.

main()
