import numpy as np
from launcher import Launcher
from animator import Animator
import yaml
import copy
import matplotlib.pyplot as plt
import multiprocessing


class GradientAscentOptimizer:

    def __init__(self):

        self.initial_params = None
        self.n_opt_params = None
        self.launcher_n_joints = None

        self.delta = None
        self.dt = None
        self.simulation_length = None

        self.min_value = 1e-3

    def optimize(self, launcher_params, n_epochs, learning_rate=0.0001, delta=1e-3, dt=0.001, simulation_length=1.5):

        self.initial_params = launcher_params
        self.launcher_n_joints = len(launcher_params['theta'])
        self.delta = delta
        self.dt = dt
        self.simulation_length = simulation_length

        params_array = self.params_dict_to_array(launcher_params)
        self.n_opt_params = len(params_array)

        E_history = np.empty(n_epochs+1)

        for i in range(n_epochs):
            gradient, E_start = self.calculate_gradient(params_array)

            E_history[i] = E_start

            params_array += learning_rate*gradient
            params_array = self.params_min_value_check(params_array)

            print(i+1)
            print(E_start)
            print()

        E_final = self.calc_E_disc(params_array)
        E_history[n_epochs] = E_final

        return E_history, self.params_array_to_dict(params_array, include_all=True)

    def calculate_gradient(self, params_array):

        params_with_delta = []
        for i in range(self.n_opt_params):
            params = np.copy(params_array)
            params[i] += self.delta
            params_with_delta.append(params)

        pool = multiprocessing.Pool(5)
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
            param_dict = {'theta': np.array([0.0, *param_array[:self.launcher_n_joints-1]]),
                          'rod_length': param_array[self.launcher_n_joints-1:2*self.launcher_n_joints-1],
                          'rod_m': param_array[2*self.launcher_n_joints-1:3*self.launcher_n_joints-1],
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


def main():
    dt = 0.001
    T = 1.5

    with open('launcher_params.yaml', 'r') as file:
        params = yaml.safe_load(file)

    optimizer = GradientAscentOptimizer()
    E_history, opt_params = optimizer.optimize(params, n_epochs=40, dt=dt, simulation_length=T)
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
