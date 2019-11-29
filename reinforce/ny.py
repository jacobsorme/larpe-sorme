import numpy as np
import random
import matplotlib.pyplot as plt
import scipy
import sys
from enum import Enum

# Some colours
LIGHT_RED    = '#FFC4CC';
LIGHT_GREEN  = '#95FD99';
BLACK        = '#000000';
WHITE        = '#FFFFFF';
LIGHT_PURPLE = '#E8D0FF';
LIGHT_ORANGE = '#FAE0C3';
LIGHT_BLUE = '#3355ee';

# Map a color to each cell in the maze
col_map = {0: WHITE, 1: BLACK, 2: LIGHT_GREEN, -6: LIGHT_RED, -1: LIGHT_RED};

maze =  np.array([
    [0,0,1,0,0,0,0,0],
    [0,0,1,0,0,1,0,0],
    [0,0,1,0,0,1,1,1],
    [0,0,1,0,0,1,0,0],
    [0,0,0,0,0,0,0,0],
    [0,1,1,1,1,1,1,0],
    [0,0,0,0,1,0,0,0]]) # (down,right)
mino_valid_moves =  np.array([
    [2,3,3,3,3,3,3,2],
    [3,4,4,4,4,4,4,3],
    [3,4,4,4,4,4,4,3],
    [3,4,4,4,4,4,4,3],
    [3,4,4,4,4,4,4,3],
    [3,4,4,4,4,4,4,3],
    [2,3,3,3,3,3,3,2]]) # the amount of valid moves from each pos.

goal = np.array([6,5])
maze_dist = []
maze_rows = 7
maze_cells = maze.size

# All possible positions (for the minotaur).
positions = np.array([[row, col] for row in range(7) for col in range(8)])
# All possible player actions.

actions = [[-1,0], [1,0], [0,-1], [0, 1], [0,0]]
# All possible minotaur actions.
MINO_STAND_STILL = True
if MINO_STAND_STILL:
    actions_mino = [[-1,0], [1,0], [0,-1], [0, 1], [0,0]]
    mino_valid_moves += 1
else:
    actions_mino = [[-1,0], [1,0], [0,-1], [0, 1]]


def main():
    global maze_dist

    do_simulate = len(sys.argv) == 2 and sys.argv[1] == 's'

    if not do_simulate:
        prob, r = calc_prob()
        print('Prob matrix calculated.')

        T = 20
        V, pi = dyn_prog(prob, r, T)
        np.save('value', V)
        np.save('policy', pi)
        np.save('reward', r)

        s0 = encode_state([0, 0], [goal[0], goal[1]])
        print(f'Probability = {V[s0, 0]}')

    else:
        V = np.load('value.npy')
        s0 = encode_state([0, 0], [goal[0], goal[1]])
        s_goal = encode_state([goal[0], goal[1]], [goal[0], goal[1]])

        print(f'Probability = {V[s0]}')
        y = [V[s0,t] for t in range(21)]

        pi = np.load('policy.npy')
        pp, mp = simulate(pi)
        animate(pp, mp)
    #maze_dist = calc_maze_dist(maze, goal)


def calc_maze_dist(maze, s):
    dist = -np.ones(maze.shape)
    calc_maze_helper(dist, s, 0)
    return dist

def calc_maze_helper(dist, s, val=0):
    if cv(s[0],s[1]):
        return
    d = dist[s[0], s[1]]
    if d != -1 and d <= val:
        return
    dist[s[0], s[1]] = val
    calc_maze_helper(dist, s + [-1,0], val + 1)
    calc_maze_helper(dist, s + [ 1,0], val + 1)
    calc_maze_helper(dist, s + [0,-1], val + 1)
    calc_maze_helper(dist, s + [0, 1], val + 1)

def encode_single_pos(pos):
    return int(pos[0] + pos[1] * maze_rows)

def encode_state(player, mino):
    return encode_single_pos(player) + maze_cells*encode_single_pos(mino)

def calc_prob():
    # p = (cur_state, action, next_state)
    n_states = maze.size * maze.size + 1
    p = np.zeros((n_states, 5, n_states))
    r = np.zeros((n_states, 5))
    #r = -np.ones((n_states, 5)) # Change here if time-critical

    for p_pos in positions: # player
        if cv(p_pos[0], p_pos[1]):
            continue

        for m_pos in positions: # minotaur
            cur_s = encode_state(p_pos, m_pos)
            mino_prob = 1 / mino_valid_moves[m_pos[0]][m_pos[1]]

            for m_act in actions_mino: # minotaur actions
                m_newpos = m_pos + m_act
                # minotaur must stay within bounds.
                if cv(m_newpos[0], m_newpos[1], True):
                    continue

                for a in range(5): # player actions
                    p_newpos = p_pos + actions[a]
                    # don't allow the player to walk into walls.
                    if cv(p_newpos[0], p_newpos[1]):
                        continue

                    if np.array_equal(p_newpos, m_newpos):
                        p[cur_s, a, -1] = mino_prob # Kill player
                    else:
                        next_s = encode_state(p_newpos, m_newpos)
                        p[cur_s, a, next_s] = mino_prob

    # Normalize each (cur_state, a) layer.
    #layer_sums = np.sum(p, (0, 1))
    #layer_sums[layer_sums == 0] = 1
    #p[:,:] /= layer_sums

    # Set transition to terminal state from goal.
    for m_pos in positions:
        if np.array_equal(goal, m_pos):
           continue

        goal_state = encode_state(goal, m_pos)
        # From goal every action leads to terminal state.
        p[goal_state,:,:]  = 0
        p[goal_state,:,-1] = 1
        r[goal_state,:] = 100

    # Terminal state only leads to terminal state.
    p[-1, :, -1] = 1
    r[-1,:] = 0 # Change here if time-critical

    return p, r

def dyn_prog(p, r, T):
    n_states  = p.shape[0]
    n_actions = p.shape[1]

    # The variables involved in the dynamic programming backwards recursions
    V      = np.zeros((n_states, T+1))
    policy = np.zeros((n_states, T+1), dtype=np.int8)

    Q            = np.copy(r)
    V[:, T]      = np.max(Q,1)
    policy[:, T] = np.argmax(Q,1)

    # The dynamic programming bakwards recursion
    for t in range(T-1,-1,-1):
        print(f't={t}')
        # Update the value function acccording to the bellman equation
        for s in range(n_states):
            for a in range(n_actions):
                # Update of the temporary Q values
                Q[s,a] = r[s,a] + np.dot(p[s,a,:],V[:,t+1])
        # Update by taking the maximum Q value w.r.t the action a
        V[:,t] = np.max(Q,1);
        # The optimal action is the one that maximizes the Q function
        policy[:,t] = np.argmax(Q,1);
    return V, policy

def simulate(pi):
    player = [(0,0)]
    mino   = [(goal[0],goal[1])]

    for t in range(20):
        # Player
        pos = player[-1]
        # best_move = [0,-999999]
        # for a in actions:
        #     new_pos = pos + a
        #     if cv(new_pos[0], new_pos[1]):
        #         continue
        #     val = -maze_dist[new_pos[0], new_pos[1]]
        #     if val > best_move[1]:
        #         best_move = [new_pos, val]

        # Player follows the policy.
        cur_state = encode_state(pos, mino[-1])
        action = actions[pi[cur_state, t]]
        player.append((pos[0] + action[0], pos[1] + action[1]))

        # Minotaur does a random walk.
        pos = np.array(mino[-1])
        free = 0
        while free == 0:
            # If mino can stand still (5 actions), stand still with 20% chance.
            if len(actions_mino) == 5 and random.random() > 0.8:
                dv, dh = 0, 0
                break
            # Elso, do a random move.
            vertical = random.randint(0, 1)
            dir      = random.randrange(-1, 2 ,2)
            dh = (1-vertical) * dir
            dv = vertical * dir
            free = 1 - cv(pos[0]+dv, pos[1]+dh, True)
        mino.append((pos[0] + dv, pos[1] + dh))

    return player, mino

def cv(v, h, minotaur=False):
    if v < 0 or v >= maze.shape[0] or h < 0 or h >= maze.shape[1]:
        return 1 # Boundary == wall
    elif minotaur:
        return 0
    return maze[v, h]

def draw_maze(show=False):
    # Give a color to each cell
    rows,cols    = maze.shape;
    colored_maze = [[col_map[maze[j,i]] for i in range(cols)] for j in range(rows)];

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows));

    # Remove the axis ticks and add title
    plt.gca().set_xticks([]);
    plt.gca().set_yticks([]);

    # Create a table to colorgrid.get_celld()[(pathP[i])]
    grid = plt.table(cellText=None, cellColours=colored_maze, cellLoc='center',
                     loc=(0,0), edges='closed');
    # Modify the hight and width of the cells in the table
    tc = grid.properties()['child_artists']
    for cell in tc:
        cell.set_height(1.0/rows);
        cell.set_width(1.0/cols);

    if show:
        plt.show()
    return grid

def animate(pathP, pathM):
    grid = draw_maze(False)
    won  = False
    dead = False

    print(pathP)

    plt.ion()
    for i in range(len(pathP)):
        # Reset previous positions
        if i > 0:
            grid.get_celld()[pathP[i-1]].set_facecolor(col_map[maze[pathP[i-1]]])
            grid.get_celld()[pathP[i-1]].get_text().set_text('')
            grid.get_celld()[pathM[i-1]].set_facecolor(col_map[maze[pathM[i-1]]])
            grid.get_celld()[pathM[i-1]].get_text().set_text('')

        # Check for victory
        if i > 0 and pathP[i-1] == (6,5):
           won = True
           grid.get_celld()[(6,5)].set_facecolor(LIGHT_GREEN)
           grid.get_celld()[(6,5)].get_text().set_text('Win')
           
        # Animate player
        if not won:
            grid.get_celld()[pathP[i]].set_facecolor(LIGHT_ORANGE)
            grid.get_celld()[pathP[i]].get_text().set_text('Player')

        # Animate minotaur
        grid.get_celld()[pathM[i]].set_facecolor(LIGHT_BLUE)
        grid.get_celld()[pathM[i]].get_text().set_text('Minotaur')
        # Check for death
        if not won and pathP[i] == pathM[i]:
            dead = True
            grid.get_celld()[pathP[i]].set_facecolor(LIGHT_RED)
            grid.get_celld()[pathP[i]].get_text().set_text('DEAD')
            plt.ioff()
            plt.show()
            break
        plt.show()
        plt.pause(0.2)
        plt.savefig(f'fig{i}')

    if not dead:
        plt.pause(3)
        plt.ioff()

main()
