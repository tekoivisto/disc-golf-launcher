from launcher import Launcher
from launcher import g_mag
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle
import numpy as np
import cv2
import yaml
from time import time


class Animator:

    def __init__(self):

        self.fig = plt.figure(figsize=(19.2/2, 10.8/2))
        self.gs = GridSpec(2, 2, figure=self.fig, width_ratios=[1, 2])
        self.ax_disc_energy = self.fig.add_subplot(self.gs[1, 0])
        self.ax_weight = self.fig.add_subplot(self.gs[0, 0])
        self.ax_launcher = self.fig.add_subplot(self.gs[:, 1])
        self.fig.subplots_adjust(wspace=0.15, hspace=0.1)

        self.ax_disc_energy.set_xlabel('time (s)')
        self.ax_disc_energy.set_ylabel('disc energy (J)')
        self.ax_launcher.set_xlabel('x (m)')
        self.ax_launcher.set_ylabel('y (m)')
        self.ax_launcher.set_aspect('equal')
        self.ax_weight.set_ylabel('height (m)')
        self.ax_weight.set_xticks([])

        self.weight_radius = 0.15
        self.weight_circle = Circle((0, 100), self.weight_radius, color='k')
        self.ax_weight.add_patch(self.weight_circle)

        rope_color = 'tab:brown'
        self.rope_extra_height = 2*self.weight_radius
        self.rope_launcher_horizontal = self.ax_launcher.plot([], [], '-', color=rope_color)[0]
        self.rope_weight_horizontal = self.ax_weight.plot([], [], '-', color=rope_color)[0]
        self.rope_pulley = self.ax_weight.plot([], [], '-', color=rope_color)[0]
        self.rope_vertical = self.ax_weight.plot([], [], '-', color=rope_color)[0]

        rubber_band_color = 'tab:green'
        self.rubber_band = self.ax_weight.plot([], [], '-', color=rubber_band_color)[0]

        self.weight_start_line = self.ax_weight.plot([], [], '--', color='gray')[0]
        self.rubber_band_start_line = self.ax_weight.plot([], [], '--', color='gray')[0]
        self.rubber_band_marker = self.ax_weight.plot([], [], 'o', color=rubber_band_color, markersize=3)[0]

        self.time_vertical_line = self.ax_disc_energy.plot([], [], '--', color='gray')[0]

        self.r_pulley = 0.04
        self.pulley = Circle((0, 100), self.r_pulley, color='gray')
        self.ax_weight.add_patch(self.pulley)

        self.winch = Circle((0, 0), 0, color='gray')
        self.ax_launcher.add_patch(self.winch)

        self.disc = Circle((0, 0), 0, color='tab:blue')
        self.ax_launcher.add_patch(self.disc)

        self.freefall_time = None
        self.launcher_lines = None
        self.launcher = None
        self.fps_video = None
        self.fps_launcher = None
        self.video = None

    def animate(self, launcher, video_fname, fps=60):

        self.launcher = launcher
        self.fps_video = fps

        self.video = self.create_video_writer(video_fname)

        n_steps = self.launcher.history['pos'].shape[0]
        timestep_jump = int(1 / fps / self.launcher.dt)
        self.fps_launcher = 1/(timestep_jump*self.launcher.dt)
        save_idx = np.arange(0, n_steps, timestep_jump)

        self.launcher_lines = [self.ax_launcher.plot([], [], 'ko-', markersize=3)[0]
                               for _ in range(self.launcher.n_joints)]
        self.launcher_lines[-1].set_marker('')
        self.disc.set_radius(self.launcher.r_disc)

        self.set_ax_lims()

        self.plot_disc_energy()

        self.draw_static_objects()

        self.animate_weight_initial_fall()

        for idx in save_idx:
            self.draw_launcher(idx)

            t = idx*self.launcher.dt + self.freefall_time
            self.time_vertical_line.set_data([t, t], [0, 1000])
            self.canvas_to_video_frame()

        self.video.release()

    def set_ax_lims(self):

        launcher_lim = 1.1 * np.sum(self.launcher.l_mag)
        self.ax_launcher.set_xlim([-launcher_lim, launcher_lim])
        self.ax_launcher.set_ylim([-launcher_lim, launcher_lim])

        weight_ylim = (self.launcher.h+self.weight_radius+self.rope_extra_height+self.r_pulley)*1.05
        self.ax_weight.set_ylim([0, weight_ylim])
        bbox = self.ax_weight.get_position()
        fig_width, fig_height = self.fig.get_size_inches()
        ratio = fig_width*bbox.width/(fig_height*bbox.height)
        weight_xlim = (weight_ylim*ratio/2)
        self.ax_weight.set_xlim([-weight_xlim, weight_xlim])

    def draw_static_objects(self):

        self.rope_launcher_horizontal.set_data([-100, self.launcher.r[0]], [self.launcher.r[1], self.launcher.r[1]])

        x_rope_pulley = np.linspace(0, self.r_pulley)
        y_rope_pulley = (self.launcher.h+self.weight_radius+self.rope_extra_height
                         + np.sqrt(self.r_pulley**2-(x_rope_pulley-self.r_pulley)**2))
        self.rope_pulley.set_data(x_rope_pulley, y_rope_pulley)

        y_hor = self.launcher.h+self.weight_radius+self.rope_extra_height+self.r_pulley
        self.rope_weight_horizontal.set_data([self.r_pulley, 10], [y_hor, y_hor])

        self.pulley.center = (self.r_pulley, y_hor-self.r_pulley)

        self.weight_start_line.set_data([-10, 10], [self.launcher.h, self.launcher.h])

        self.rubber_band_start_line.set_data([-10, 10],
                                             [self.launcher.h_rubber_band_initial, self.launcher.h_rubber_band_initial])

        self.winch.set_radius(self.launcher.r_mag)

    def create_video_writer(self, video_fname):

        self.fig.canvas.draw()
        img = self.fig.canvas.renderer.buffer_rgba()
        height = img.shape[0]
        width = img.shape[1]

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        return cv2.VideoWriter(video_fname, fourcc, self.fps_video, (width, height))

    def draw_launcher(self, history_idx, draw_weight=True):

        release_idx = self.launcher.disc_release_timestep
        if history_idx <= release_idx:
            pos = self.launcher.history['pos'][history_idx]
            theta = self.launcher.history['theta'][history_idx]
            pos_disc = self.launcher.history['pos disc'][history_idx]
        else:
            pos = self.launcher.history_disc_released['pos'][history_idx-release_idx]
            theta = self.launcher.history_disc_released['theta'][history_idx-release_idx]
            pos_disc = self.launcher.history_disc_released['pos disc'][history_idx-release_idx]

        if draw_weight:
            h_weight = self.launcher.history['h weight'][history_idx]
            h_rubber_band = self.launcher.history['h rubber band'][history_idx]
            weight_start_y = self.launcher.h + self.weight_radius
            weight_y = h_weight + self.weight_radius

            if h_weight >= h_rubber_band:
                self.rope_vertical.set_data([0, 0], [weight_start_y+self.rope_extra_height, weight_y])
                self.rubber_band.set_data([], [])
                self.rubber_band_marker.set_data([], [])

            else:
                rubber_band_top_y = weight_start_y + h_rubber_band - self.launcher.h
                self.rope_vertical.set_data([0, 0], [weight_start_y+self.rope_extra_height, rubber_band_top_y])
                self.rubber_band.set_data([0, 0], [rubber_band_top_y, weight_y])
                self.rubber_band_marker.set_data([0], [rubber_band_top_y])

            self.weight_circle.center = (0, weight_y)

        for i in range(self.launcher.n_joints):
            if i == self.launcher.n_joints-1:
                l_mag = self.launcher.l_mag_last_initial
                if history_idx <= release_idx:
                    c_mag = self.launcher.c_mag_last_with_disc
                else:
                    c_mag = self.launcher.c_mag_last_initial
            else:
                l_mag = self.launcher.l_mag[i]
                c_mag = self.launcher.c_mag[i]

            unit_vec = np.array([np.cos(theta[i]), np.sin(theta[i])])

            start = pos[i] - c_mag * unit_vec
            end = pos[i] + (l_mag - c_mag) * unit_vec

            self.launcher_lines[i].set_data([start[0], end[0]], [[start[1], end[1]]])

        self.disc.center = pos_disc

    def animate_weight_initial_fall(self):

        self.draw_launcher(0, draw_weight=False)

        t_fall_steps = np.arange(0, self.freefall_time, 1/self.fps_launcher)
        delta_last = self.freefall_time-t_fall_steps[-1]
        t_fall_steps += delta_last
        t_fall_steps[1:] = t_fall_steps[:-1]
        t_fall_steps[0] = 0

        weight_start_y = self.launcher.h + self.weight_radius
        for t_f in t_fall_steps:
            weight_y = weight_start_y - 0.5 * g_mag * t_f ** 2

            self.rope_vertical.set_data([0, 0], [weight_start_y+self.rope_extra_height, weight_y])

            self.weight_circle.center = (0, weight_y)

            if t_f != 0:
                self.time_vertical_line.set_data([t_f, t_f], [0, 1000])

            self.canvas_to_video_frame()

    def canvas_to_video_frame(self):
        self.fig.canvas.draw()
        img = np.array(self.fig.canvas.renderer.buffer_rgba())
        img = cv2.cvtColor(img[:, :, :-1], cv2.COLOR_RGB2BGR)
        self.video.write(img)

    def plot_disc_energy(self):

        release_timestep = self.launcher.disc_release_timestep

        E_tot, E_trans, E_rot = self.launcher.calculate_disc_energy(self.launcher.history)
        E_tot_released, E_trans_released, E_rot_released = self.launcher.calculate_disc_energy(self.launcher.history_disc_released)

        self.freefall_time = np.sqrt(2*(self.launcher.h-self.launcher.h_rubber_band_initial)/g_mag)
        fall_n_steps = int(self.freefall_time/self.launcher.dt)
        n_steps = E_trans.shape[0] + fall_n_steps
        t = np.linspace(0, n_steps*self.launcher.dt, n_steps)

        first_zeros = np.zeros(fall_n_steps)
        E_trans = np.hstack((first_zeros, E_trans))
        E_rot = np.hstack((first_zeros, E_rot))
        E_tot = np.hstack((first_zeros, E_tot))

        v_mag_max = np.linalg.norm(self.launcher.history['v disc'][release_timestep])

        release_timestep += fall_n_steps

        self.ax_disc_energy.set_xlim([t[0], t[-1]-1/self.fps_launcher])
        self.ax_disc_energy.set_ylim([0, 1.1 * E_tot[release_timestep]])
        # self.ax_disc_energy.plot(t, np.array([E_tot, E_trans, E_rot]).T)
        self.ax_disc_energy.plot(t[:release_timestep], E_tot[:release_timestep], color='tab:blue')
        self.ax_disc_energy.plot(t[release_timestep:], E_tot[release_timestep:], '--', color='tab:blue')
        self.ax_disc_energy.plot(t[release_timestep:], E_tot_released, color='tab:blue')

        self.ax_disc_energy.annotate(f'max velocity:\n{v_mag_max*3.6:.1f} km/h',
                                     xy=(t[release_timestep], E_tot[release_timestep]), xytext=(0.1, 0.8*E_tot[release_timestep]),
                                     arrowprops={'facecolor': 'black', 'width': 0.1, 'headwidth': 5, 'headlength': 7.5,
                                                 'shrink': 0.15}, fontsize=8)


def main():

    with open('launcher_params.yaml', 'r') as file:
        params = yaml.safe_load(file)

    dt = 0.0001
    T = 1.5

    launcher = Launcher(params)
    start_time = time()
    launcher.simulate(dt, T)

    print(f'simulation done in {time()-start_time:.3f} s\nanimating video')

    animator = Animator()
    animator.animate(launcher, 'animations/video.mp4')


if __name__ == '__main__':
    main()
