import numpy as np
from launcher import Launcher
from animator import Animator
import yaml
import copy
import matplotlib.pyplot as plt
import multiprocessing
import os
import pandas as pd


class GradientAscentOptimizer:

    def __init__(self):

        self.initial_params = None
        self.n_opt_params = None
        self.launcher_n_rods = None

        self.learning_rate = None
        self.delta = None
        self.dt = None
        self.simulation_length = None

        self.min_value = 1e-3

        self.save_dir = None
        self.opt_history = {}

    def optimize(self, launcher_params, n_epochs, learning_rate=0.0002, delta=1e-04, dt=0.0001, simulation_length=1.5,
                 save_dir='optimization_run'):

        self.initial_params = launcher_params
        self.launcher_n_rods = len(launcher_params['rod_length'])
        self.learning_rate = learning_rate
        self.delta = delta
        self.dt = dt
        self.simulation_length = simulation_length

        params_array = self.params_dict_to_array(launcher_params)
        self.n_opt_params = len(params_array)

        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.init_opt_history(n_epochs)

        E_history = np.empty(n_epochs+1)

        for i in range(n_epochs):

            gradient, E_start = self.calculate_gradient(params_array)

            print(f'epoch {i}')
            print(f'disc energy: {E_start:.3f} J')
            print()

            E_history[i] = E_start
            self.save_epoch(params_array, E_start, i)

            params_array_new = params_array + self.learning_rate * gradient

            params_array = self.params_max_change_check(params_array, params_array_new)
            params_array = self.params_min_value_check(params_array)

        E_final = self.calc_E_disc(params_array)
        E_history[n_epochs] = E_final
        self.save_epoch(params_array, E_final, n_epochs)
        self.save_opt_history()

        print(f'epoch {n_epochs}')
        print(f'disc energy: {E_final:.3f} J')
        print()

        return E_history, self.params_array_to_dict(params_array, include_all=True)

    def calculate_gradient(self, params_array):

        params_with_delta = []
        for i in range(self.n_opt_params):
            params = np.copy(params_array)

            params[i] += self.delta
            params_with_delta.append(params)

        pool = multiprocessing.Pool(8)
        E_all = pool.map(self.calc_E_disc, [params_array]+params_with_delta)
        pool.close()
        pool.join()

        E_start = E_all[0]
        E_with_delta = np.array(E_all[1:], dtype=float)

        delta_E = E_with_delta - E_start
        gradient = delta_E / self.delta

        return gradient, E_start

    def calc_E_disc(self, param_array):

        launcher = self.params_array_to_launcher(param_array)
        E_disc = launcher.simulate(self.dt, self.simulation_length, release_disc=False)
        return E_disc

    @staticmethod
    def params_dict_to_array(param_dict):

        array = np.array([*param_dict['theta'][1:], *param_dict['rod_length'], *param_dict['rod_m'],
                          param_dict['rubber_band_k'], param_dict['rubber_band_h'], param_dict['winch_r']])

        return array

    def params_array_to_dict(self, param_array, include_all=False):

        if not include_all:
            param_dict = {'theta': np.array([0.0, *param_array[:self.launcher_n_rods]]),
                          'rod_length': param_array[self.launcher_n_rods:2*self.launcher_n_rods],
                          'rod_m': param_array[2*self.launcher_n_rods:3*self.launcher_n_rods],
                          'rubber_band_k': param_array[-3], 'rubber_band_h': param_array[-2],
                          'winch_r': param_array[-1]}
            return param_dict

        else:
            all_params = copy.deepcopy(self.initial_params)

            opt_params_dict = self.params_array_to_dict(param_array)

            for key in opt_params_dict:
                all_params[key] = opt_params_dict[key]

            return all_params

    def params_array_to_launcher(self, opt_params):

        param_dict = self.params_array_to_dict(opt_params, include_all=True)

        launcher = Launcher(param_dict)

        return launcher

    def params_min_value_check(self, param_array):
        param_dict = self.params_array_to_dict(param_array)

        for key in ['rod_length', 'rod_m', 'rubber_band_k', 'winch_r']:
            if isinstance(param_dict[key], np.ndarray):
                param_dict[key][param_dict[key] < self.min_value] = self.min_value
            else:
                if param_dict[key] < self.min_value:
                    param_dict[key] = self.min_value

        param_array = self.params_dict_to_array(param_dict)

        return param_array

    def params_max_change_check(self, old_params_array, new_params_array):
        """Limits the maximum change in a parameter during a single epoch. The derivative of disc energy is
        discontinuous due weight hitting ground. The numerical gradients are occasionally very large, possibly caused by
        the discontinuity."""

        max_prop_change = 100 * self.learning_rate

        idx_positive = (new_params_array-old_params_array)/old_params_array > max_prop_change
        idx_negative = (old_params_array-new_params_array)/old_params_array > max_prop_change

        new_params_array[idx_positive] = old_params_array[idx_positive]*(1+max_prop_change)
        new_params_array[idx_negative] = old_params_array[idx_negative]*(1-max_prop_change)

        return new_params_array

    def init_opt_history(self, n_epochs):

        for key in ['theta', 'rod_length', 'rod_m']:
            for i in range(self.launcher_n_rods+1):
                if i == self.launcher_n_rods and key != 'theta':
                    continue
                self.opt_history[f'{key}_{i}'] = np.zeros(n_epochs+1)

        for key in [f'theta_{self.launcher_n_rods}', 'rubber_band_k', 'rubber_band_h', 'winch_r', 'E_disc']:
            self.opt_history[key] = np.zeros(n_epochs+1)

    def save_epoch(self, params_array, E_disc, epoch):

        params_dict = self.params_array_to_dict(params_array)

        for key in ['rubber_band_k', 'rubber_band_h', 'winch_r']:
            self.opt_history[key][epoch] = params_dict[key]

        self.opt_history['E_disc'][epoch] = E_disc

        for i in range(self.launcher_n_rods):
            for key in ['theta', 'rod_length', 'rod_m']:
                self.opt_history[f'{key}_{i}'][epoch] = params_dict[key][i]
        self.opt_history[f'theta_{self.launcher_n_rods}'][epoch] = params_dict['theta'][self.launcher_n_rods]

    def save_opt_history(self):

        opt_history_df = pd.DataFrame.from_dict(self.opt_history)
        opt_history_df.to_csv(f'{self.save_dir}/opt_history.csv')


def main():
    dt = 0.0001
    T = 1

    with open('launcher_params.yaml', 'r') as file:
        params = yaml.safe_load(file)

    optimizer = GradientAscentOptimizer()
    E_history, opt_params = optimizer.optimize(params, n_epochs=200, dt=dt, simulation_length=T)
    print(opt_params)

    launcher = Launcher(opt_params)
    launcher.simulate(dt, T)

    animator = Animator()
    animator.animate(launcher, 'animations/optimized_launcher.mp4')

    plt.figure()
    plt.plot(E_history)
    plt.xlabel('epoch')
    plt.ylabel('disc energy (J)')

    plt.show()


if __name__ == '__main__':
    main()
