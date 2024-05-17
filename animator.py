from launcher import Launcher
from launcher import g_mag
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle
import numpy as np
import cv2


class Animator:

    def __init__(self):

        self.fig = plt.figure(figsize=(19.2/2, 10.8/2))
        self.gs = GridSpec(1, 2, figure=self.fig, width_ratios=[1, 2])
        self.ax_weight = self.fig.add_subplot(self.gs[0, 0])
        self.ax_launcher = self.fig.add_subplot(self.gs[0, 1])

        self.ax_launcher.set_xlabel('x (m)')
        self.ax_launcher.set_ylabel('y (m)')
        self.ax_launcher.set_aspect('equal')
        self.ax_weight.set_ylabel('height (m)')
        self.ax_weight.set_xticks([])

        self.weight_radius = 0.1
        self.weight_circle = Circle((0, 100), self.weight_radius, color='k')
        self.ax_weight.add_patch(self.weight_circle)

        rope_color = 'tab:brown'
        self.rope_extra_height = 3*self.weight_radius
        self.rope_launcher_horizontal = self.ax_launcher.plot([], [], '-', color=rope_color)[0]
        self.rope_weight_horizontal = self.ax_weight.plot([], [], '-', color=rope_color)[0]
        self.rope_pulley = self.ax_weight.plot([], [], '-', color=rope_color)[0]
        self.rope_vertical = self.ax_weight.plot([], [], '-', color=rope_color)[0]

        rubber_band_color = 'tab:green'
        self.rubber_band = self.ax_weight.plot([], [], '-', color=rubber_band_color)[0]

        self.weight_start_line = self.ax_weight.plot([], [], '--', color='gray')[0]
        self.rubber_band_start_line = self.ax_weight.plot([], [], '--', color='gray')[0]
        self.rubber_band_marker = self.ax_weight.plot([], [], 'o', color=rubber_band_color, markersize=3)[0]

        self.r_pulley = 0.025
        self.pulley = Circle((0, 100), self.r_pulley, color='gray')
        self.ax_weight.add_patch(self.pulley)

        self.winch = Circle((0, 0), 0, color='gray')
        self.ax_launcher.add_patch(self.winch)

        self.launcher_lines = None
        self.launcher = None
        self.fps = None
        self.video = None

    def animate(self, launcher, video_fname, fps=30):

        self.launcher = launcher
        self.fps = fps

        self.video = self.create_video_writer(video_fname)

        n_steps = self.launcher.pos_history.shape[0]
        frame_jump = int(1 / fps / self.launcher.dt)
        save_idx = np.array(range(0, n_steps, frame_jump))

        self.launcher_lines = [self.ax_launcher.plot([], [], 'ko-', markersize=3)[0]
                               for _ in range(self.launcher.n_joints)]
        self.launcher_lines[-1].set_marker('')

        self.set_ax_lims()

        self.draw_static_objects()

        self.animate_weight_initial_fall()

        for idx in save_idx:
            self.draw_launcher(idx)

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

        return cv2.VideoWriter(video_fname, fourcc, self.fps, (width, height))

    def draw_launcher(self, history_idx, draw_weight=True, save_to_video=True):

        pos = self.launcher.pos_history[history_idx]
        theta = self.launcher.theta_history[history_idx]

        if draw_weight:
            h_weight = self.launcher.h_weight_history[history_idx]
            h_rubber_band = self.launcher.h_rubber_band_history[history_idx]
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
            c_mag = self.launcher.c_mag[i]
            l_mag = self.launcher.l_mag[i]

            unit_vec = np.array([np.cos(theta[i]), np.sin(theta[i])])

            start = pos[i] - c_mag * unit_vec
            end = pos[i] + (l_mag - c_mag) * unit_vec

            self.launcher_lines[i].set_data([start[0], end[0]], [[start[1], end[1]]])

        if save_to_video:
            self.canvas_to_video_frame()

    def animate_weight_initial_fall(self):

        self.draw_launcher(0, draw_weight=False, save_to_video=False)

        t_fall = np.sqrt(2*(self.launcher.h-self.launcher.h_rubber_band_initial)/g_mag)
        t_fall_steps = np.arange(0, t_fall, 1/self.fps)

        weight_start_y = self.launcher.h + self.weight_radius
        for t_f in t_fall_steps:
            weight_y = weight_start_y - 0.5 * g_mag * t_f ** 2

            self.rope_vertical.set_data([0, 0], [weight_start_y+self.rope_extra_height, weight_y])

            self.weight_circle.center = (0, weight_y)

            self.canvas_to_video_frame()

    def canvas_to_video_frame(self):
        self.fig.canvas.draw()
        img = np.array(self.fig.canvas.renderer.buffer_rgba())
        img = cv2.cvtColor(img[:, :, :-1], cv2.COLOR_RGB2BGR)
        self.video.write(img)


def main():

    n_rods = 4
    theta = np.full(n_rods, -np.pi/2)
    l_mag = np.ones(n_rods)
    c_mag = np.full(n_rods, 0.5)
    m = np.ones(n_rods)
    I = np.ones(n_rods)
    r_mag = 0.3
    m_weight = 10.0
    h = 1.0
    h_rubber_band = 0.5
    k = 1000.0

    dt = 0.001
    T = 5

    launcher = Launcher(theta, l_mag, c_mag, m, I, r_mag, m_weight, h, h_rubber_band, k)
    launcher.simulate(dt, T)

    print('simulation done, animating video')

    animator = Animator()
    animator.animate(launcher, 'animations/video.mp4')


if __name__ == '__main__':
    main()
