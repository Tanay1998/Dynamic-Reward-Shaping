import gym
import numpy as np
import sys
from gym.envs.toy_text import discrete

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3
DELTAS = [(0, 1), (1, 0), (0, -1), (-1, 0)]
class MutliGridworldEnv(discrete.DiscreteEnv):

    metadata = {'render.modes': ['human', 'ansi']}

    def get_actions(self, action_num):
        actions = []
        for i in range(self.num_agents):
            actions.append(action_num % self.num_actions)
            action_num = int(action_num / self.num_actions)
        return actions

    def get_state(self, positions):
        state = 0
        size = np.prod(self.shape)
        mult = 1
        for pos in positions:
            state += np.ravel_multi_index(tuple(pos), self.shape) * mult
            mult *= size
        return state

    def get_positions(self, state):
        positions = []
        size = int(np.prod(self.shape))
        for i in range(self.num_agents):
            positions.append(np.unravel_index(state % size, self.shape))
            state = int(state / size)
        return positions

    def _limit_coordinates(self, coord):
        coord[0] = min(coord[0], self.shape[0] - 1)
        coord[0] = max(coord[0], 0)
        coord[1] = min(coord[1], self.shape[1] - 1)
        coord[1] = max(coord[1], 0)
        return coord

    def distance_from_goals(self, new_position):
        D = np.array([new_position]) - np.array(self.goals)
        return np.sum([np.linalg.norm(x) for x in D])

    def distance_pairs(self, positions):
        D = 0
        collisions = 0
        for i, pos1 in enumerate(positions):
            for j, pos2 in enumerate(positions):
                if i > j:
                    d = np.array(pos1) - np.array(pos2)
                    if np.dot(d, d) < 2:
                        collisions += 1
                    D += 1 / (1e-3 + 0.1 * np.dot(d, d))
        return collisions, D

    
    def _calculate_transition_prob(self, current, deltas):
        L = 0.1
        PUNISH = -1e5
        X = 5.0
        landmarkdist = 0
        new_positions = []
        is_done = True
        for pos, delta in zip(current, deltas): 
            new_position = np.array(pos) + np.array(delta)
            new_position = self._limit_coordinates(new_position).astype(int)
            new_position = tuple(new_position.astype(np.int8).tolist())
            new_positions.append(new_position)
            landmarkdist += (-L * self.distance_from_goals(new_position))
        is_done = set(new_positions) == set(self.goals)

        collisions, pairwisedist = self.distance_pairs(new_positions)
        pairwisedist = (-X * pairwisedist)

        step = 500 if is_done else -1
        step += collisions * PUNISH

        reward = (landmarkdist, pairwisedist, step)
        new_state = self.get_state(new_positions)
        return [(1.0, new_state, reward, is_done)]

    def __init__(self):

        self.shape = (10, 10)
        self.goals = [(3, 7), (1, 9)]
        self.num_goals = len(self.goals)
        self.num_agents = self.num_goals
        self.num_actions = 4
        nS = np.prod(self.shape) ** self.num_agents
        nA = self.num_actions ** self.num_agents

        # Calculate transition probabilities
        P = {} # {s: {a: s'}}
        for s in range(nS):
            positions = self.get_positions(s)
            P[s] = { a : [] for a in range(nA) }
            for a in range(nA):
                actions = self.get_actions(a)
                deltas = [DELTAS[x] for x in actions]
                P[s][a] = self._calculate_transition_prob(positions, deltas)

        isd = np.ones(nS) / (1. * nS)

        super(MutliGridworldEnv, self).__init__(nS, nA, P, isd)

    def _render(self, mode='human', close=False):
        if close:
            return

        outfile = StringIO() if mode == 'ansi' else sys.stdout
        positions = self.get_positions(self.s)
        size = np.prod(self.shape)
        for s in range(size):
            position = np.unravel_index(s, self.shape)
            # print(self.s)
            if position in positions:
                output = " x "
            elif position in self.goals:
                output = " T "
            else:
                output = " o "

            if position[1] == 0:
                output = output.lstrip()
            if position[1] == self.shape[1] - 1:
                output = output.rstrip()
                output += "\n"

            outfile.write(output)
        outfile.write("\n")
