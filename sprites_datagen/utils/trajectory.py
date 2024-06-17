########################################################################################
# Adapted from (https://github.com/akosiorek/sqair/blob/master/sqair/data/trajectory.py):
# Sequential Attend, Infer, Repeat (SQAIR)
# Copyright (C) 2018  Adam R. Kosiorek, Oxford Robotics Institute and
#     Department of Statistics, University of Oxford
#
# email:   adamk@robots.ox.ac.uk
# webpage: http://akosiorek.github.io/
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#
########################################################################################

import abc
import numpy as np


class Trajectory(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, n_dim, n_state, bounds=None):
        super(Trajectory, self).__init__()

        self._n_dim = n_dim
        self._n_state = n_state

        if bounds is not None:
            bounds = np.asarray(bounds)
            assert bounds.ndim == 2
            assert bounds.shape[0] == self._n_state
            assert bounds.shape[1] == 2

        self._bounds = bounds

    @abc.abstractmethod
    def _forward(self, state):
        """Propagates the state forward
        :param state:
        :return: (point, new_state) tuple
        """
        return None

    def _init(self, n_trajectories):
        """Initializes the trajectory.
        The default initialisation is uniform [0, 1] if no bounds are given and
        [min, max] if bounds are given. The created state is forward-propagated
        once to create an initial point.
        :return: state"""

        state = np.random.uniform(size=(n_trajectories, self._n_state))
        # state = (state - 0.5) * 2

        if self._bounds is not None:
            min_value = self._bounds[np.newaxis, :, 0]
            max_value = self._bounds[np.newaxis, :, 1]
            span = max_value - min_value

            state = min_value + state * span

        return self.forward(state)

    def _clip(self, state):
        return np.clip(state, self._bounds[:, 0], self._bounds[:, 1])

    def forward(self, state):
        state = self._clip(self._forward(state))
        return state[:, :self._n_dim].copy(), state

    def create(self, n_timesteps, n_trajectories=1, with_presence=False, init_from=None):
        """Create `n_trajectories` trajectories, which are `n_timesteps` long
        :param n_timesteps: int, length of every trajectory
        :param n_trajectories: int, number of trajectories
        :return: np.ndarray of shape (n_timesteps, n_trajectories, self.n_dim)
        """

        tjs = np.empty((n_timesteps, n_trajectories, self._n_dim), dtype=np.float32)
        tjs[0], state = self._init(n_trajectories)

        if init_from is not None:
            tjs[0] = init_from[:self._n_dim]
            state[:, :self._n_state] = init_from.copy()

        for t in range(1, n_timesteps):
            tjs[t], state = self.forward(state)

        if with_presence:
            smaller = np.less(tjs, self._bounds[:self._n_dim, 0])
            greater = np.greater(tjs, self._bounds[:self._n_dim, 1])
            presence = np.logical_not(np.logical_or(smaller, greater))
            tjs = tjs, presence.astype(np.uint8)
        return tjs


class ConstantSpeedTrajectory(Trajectory):

    def __init__(self, n_dim, pos_bounds, max_speed, bounce=True):
        self._bounce = bounce
        bounds = list(pos_bounds) + [[-max_speed, max_speed]] * 2

        super(ConstantSpeedTrajectory, self).__init__(n_dim, 2 * n_dim, bounds)

    def _forward(self, state):
        """Assume that state is [position, velocity]
        :param state:
        :return:
        """
        pos, vel = np.split(state, 2, -1)

        pos += vel

        if self._bounce:
            for d in range(self._n_dim):
                too_small = np.less(pos[:, d], self._bounds[d, 0])
                too_big = np.greater(pos[:, d], self._bounds[d, 1])

                pos[too_small, d] = 2 * self._bounds[d, 0] - pos[too_small, d]
                pos[too_big, d] = 2 * self._bounds[d, 1] - pos[too_big, d]
                vel[np.logical_or(too_small, too_big), d] *= -1

        state = np.concatenate((pos, vel), -1)
        return state

