import numpy as np
import matplotlib.pyplot as plt
from enum import Enum

city =  np.array([
    [1,0,0,0,0,1],
    [0,0,0,0,0,0],
    [1,0,0,0,0,1]])

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
    # If True (1), plot V[s0] for different lambdas.
    PLOT_DIFFERENT_LAMBDAS = 0
    # .. otherwise, use lambda below and simulate game.
    LAMBDA = 0.7

    lambdas =  [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
    lambdas = np.linspace(0.01, 0.99, 100)
    V0_last = []

    p, r = calc_prob()
    epsilon = 0.00000001
    N = 500
    for l in lambdas:
        lamb = l if PLOT_DIFFERENT_LAMBDAS else LAMBDA
        V, policy, V0_hist = value_iteration(p, r, lamb, epsilon, N)
        V0_last.append(V0_hist[-1])

        if not PLOT_DIFFERENT_LAMBDAS:
            break

    if PLOT_DIFFERENT_LAMBDAS:
        plt.figure(figsize=(8, 3.5))
        plt.plot(lambdas, V0_last)
        plt.title("Initial state value for different discount rates")
        plt.xlabel("Discount Rate (lambda)")
        plt.ylabel("V[s0]")
        plt.xticks(np.linspace(0, 1, 11))
        plt.tight_layout()
        #plt.savefig("different_lambdas")
        plt.show()
    else:
        T = 100
        hist, reward = simulate(p, r, policy, T)
        animate(hist)

def create_policy(Q):
    return np.argmax(Q, axis=1) # Return S x 1 vector.

def get_state_grid(pos):
    return int(pos[0] + pos[1] * 3)

def get_state(thief,cop):
    return get_state_grid(thief) + 18 * get_state_grid(cop)

def cv(p):
    return p[0] < 0 or p[0] >= 3 or p[1] < 0 or p[1] >= 6

def calc_prob():
    # p = (cur_state, action, next_state)
    p = np.zeros((n_states, n_actions, n_states))
    r = np.zeros((n_states, n_actions))

    init_s = get_state(pos0, pol_station)

    for p_pos in positions: # player
        for c_pos in positions: # cop
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
                    if abs(new_pos[1] - p_pos[1]) > dydx[1]:
                        continue
                elif same_x:
                    if abs(new_pos[0] - p_pos[0]) > dydx[0]:
                        continue
                elif np.sum(np.abs(new_pos - p_pos)) > manhattan_d:
                    continue

                valid_actions.append(a)
            cop_prob = 1 / len(valid_actions)

            for c_act in valid_actions: # cop actions
                c_newpos = c_pos + c_act
                for p_act in range(n_actions): # player actions
                    p_newpos = p_pos + actions[p_act]
                    if not cv(p_newpos):
                        p[cur_s, p_act, get_state(p_newpos, c_newpos)] = cop_prob
    return p, r

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
        #print(np.linalg.norm(V - best_V))
        V0_hist[n-1] = best_V[init_s]

    # Compute policy
    policy = np.argmax(Q,1)
    print(f'Value iteration done, V0={best_V[init_s]}')
    return V, policy, V0_hist[:n]

def simulate(p, r, policy, T):
    history = np.zeros((T+1), dtype=int)
    history[0] = get_state(pos0, pol_station) # initial state
    sum_reward = 0

    # Simulate and store sequence of states, in a general fashion.
    for t in range(1, T + 1):
        s = history[t-1]
        a = policy[s] # Get best action from pi(s).
        sum_reward += r[s, a] # Add reward from r(s, a).

        # Randomly pick one of the possible states from p(s+1 | s, a).
        possible_s = (np.array(np.nonzero(p[s, a]))).flatten()
        next_s = np.random.choice(possible_s)
        history[t] = next_s

    print('Simulation total reward: ' + str(sum_reward))
    return history, sum_reward

def draw_city():
    # Give a color to each cell
    rows,cols    = city.shape[0], city.shape[1]
    colors = [['#ffffff' if city[j,i] == 0 else '#aaaaff' for i in range(cols)] for j in range(rows)]

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows));
    # Remove the axis ticks and add title
    plt.gca().set_xticks([]);
    plt.gca().set_yticks([]);

    # Create a table to colorgrid.get_celld()[(pathP[i])]
    grid = plt.table(cellText=None, cellColours=colors, cellLoc='center',
                     loc=(0,0), edges='closed');
    # Modify the hight and width of the cells in the table
    tc = grid.properties()['child_artists']
    for cell in tc:
        cell.set_height(1.0/rows);
        cell.set_width(1.0/cols);

    return grid, colors

def row_col(s, cop=False):
    ''' Decodes the state, returning row and col. If cop is true, extract
    the cop's position instead.
    '''
    if cop:
        s = s // 18 # Get the cop's flattened state.
    else:
        s = s  % 18 # Remove cop from encoded state.
    return s % 3, s // 3 # there are 3 rows.

def animate(hist):
    grid, colors = draw_city()
    T = hist.shape[0]

    plt.ion()
    for t in range(T):
        # Reset map grid.
        for row in range(city.shape[0]):
            for col in range(city.shape[1]):
                grid.get_celld()[(row, col)].set_facecolor(colors[row][col])
                grid.get_celld()[(row, col)].get_text().set_text('')

        # Animate player
        row, col = row_col(hist[t])
        grid.get_celld()[(row, col)].get_text().set_text('Player')

        # Animate police
        prow, pcol = row_col(hist[t], True)
        grid.get_celld()[(prow, pcol)].get_text().set_text('Cop')

        if row == prow and col == pcol:
            grid.get_celld()[(row, col)].set_facecolor('#aa3355')

        plt.show()
        plt.pause(0.4)
    plt.ioff()
    plt.pause(1)
    plt.show() # Keep graph alive.

main()
