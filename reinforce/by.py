import numpy as np
import matplotlib.pyplot as plt
from enum import Enum

actions = np.array([[0,0], [1,0], [-1,0], [0,1], [0,-1]], dtype=int)

mapsize = 4
n_states = pow((mapsize * mapsize), 2)

def ass():
    epsilon = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]
    epsilon = list(map(lambda x: "Epsilon: "+str(x),epsilon))
    files = [   "sarsa_100000_100_0.01.npy",
                "sarsa_100000_100_0.05.npy",
                "sarsa_100000_100_0.1.npy",
                "sarsa_100000_100_0.2.npy",
                "sarsa_100000_100_0.3.npy",
                "sarsa_100000_100_0.5.npy"
                ]
    for f in files:
        plt.plot(np.load(f))
    plt.legend(epsilon)
    plt.title("T = 100")
    plt.ylabel("V[s0]")
    plt.xlabel("Episodes")
    plt.show()


def main():
    ass()
    exit()
    K = 100000
    T = 100
    epsilon = 0.01

    #Q_star, V_hist = Qlearning(K, T, 0.8)
    Q_star, V_hist = SARSA(K, T, 0.8, epsilon)

    np.save(f'sarsa_{K}_{T}_{epsilon}', V_hist)
    plt.plot(V_hist)
    plt.show()

    hist, reward = simulate(Q_star, T)
    print('Total reward: ' + str(reward))
    animate(hist)

def create_policy(Q):
    return np.argmax(Q, axis=1) # Return S x 1 vector.

def get_state_grid(pos):
    return int(pos[0] + pos[1] * mapsize)


def get_state(thief,cop):
    return get_state_grid(thief) + 16*get_state_grid(cop)

def move(pos, delta):
    new = pos + delta
    if new[0] < 0 or new[0] >= mapsize or new[1] < 0 or new[1] >= mapsize:
        return pos
    return new

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

def Qlearning(K, T, lamb):
    Q = np.zeros((n_states, actions.shape[0]))
    V_hist = np.zeros(K)
    times_visited = np.copy(Q)

    for k in range(K):
        h, _ = simulate(Q, T, 1) # h (row t): [state_t, action_t, reward_t]

        for t in range(T):
            s = h[t, 0]
            next_s = h[t+1, 0]
            a = h[t, 1]
            times_visited[s, a] += 1
            alpha = 1 / (times_visited[s, a] + 1)

            Q[s,a] += alpha * (h[t, 2] + lamb * np.max(Q[next_s]) - Q[s,a])

        V_hist[k] = np.max(Q[240,:]) # 240 is initial state.

    return Q, V_hist

def SARSA(K, T, lamb, epsilon):
    #Q = np.random.rand(n_states, actions.shape[0])
    Q = np.zeros((n_states, actions.shape[0])) # Set to zero doesn't work.
    V_hist = np.zeros(K)
    times_visited = np.copy(Q)

    for k in range(K):
        alpha = 1 / (k + 1) # Step sizes

        h, _ = simulate(Q, T, epsilon) # h (row t): [state_t, action_t, reward_t]

        for t in range(T):
            s = h[t, 0]
            next_s = h[t+1, 0]
            a = h[t, 1]
            next_a = h[t+1, 1]
            times_visited[s, a] += 1
            alpha = 1 / (times_visited[s, a] + 1)

            Q[s,a] += alpha * (h[t, 2] + lamb * Q[next_s,next_a] - Q[s,a])

        V_hist[k] = np.max(Q[0,:])

    return Q, V_hist


def draw_city():
    # Give a color to each cell
    rows,cols    = mapsize, mapsize
    colored_city = [['#ffffff' for i in range(cols)] for j in range(rows)]

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
