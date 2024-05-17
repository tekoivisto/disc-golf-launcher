from launcher import Launcher
from launcher import g_mag
import matplotlib.pyplot as plt
import numpy as np
import cv2


class Animator:

    def __init__(self):

        self.fig, self.ax = plt.subplots(1, 1)
        self.ax.set_xlabel('x (m)')
        self.ax.set_ylabel('y (m)')
        self.ax.set_aspect('equal')

        self.weight_point = self.ax.plot([], [], 'ko')[0]
        rope_color = 'tab:brown'
        self.rope_horizontal = self.ax.plot([], [], '-', color=rope_color)[0]
        self.rope_vertical = self.ax.plot([], [], '-', color=rope_color)[0]
        self.rubber_band = self.ax.plot([], [], '-', color='tab:green')[0]
        self.rod_lines = None

        self.launcher = None
        self.weight_x = None
        self.fps = None
        self.video = None

    def animate(self, launcher, video_fname, fps=30, lim=6, weight_x=-3):

        self.launcher = launcher
        self.fps = fps
        self.weight_x = weight_x

        n_steps = self.launcher.pos_history.shape[0]

        frame_jump = int(1 / fps / self.launcher.dt)
        save_idx = np.array(range(0, n_steps, frame_jump))

        self.rod_lines = [self.ax.plot([], [], 'ko-')[0] for i in range(self.launcher.n_joints)]
        self.rope_horizontal.set_data([self.weight_x, self.launcher.r[0]], [self.launcher.r[1], self.launcher.r[1]])

        self.ax.set_xlim([-lim, lim])
        self.ax.set_ylim([-lim, lim])

        self.fig.canvas.draw()
        img = self.fig.canvas.renderer.buffer_rgba()
        height = img.shape[0]
        width = img.shape[1]

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video = cv2.VideoWriter(video_fname, fourcc, self.fps, (width, height))

        self.animate_weight_initial_fall()

        for idx in save_idx:
            self.draw_launcher(idx)

        self.video.release()

    def draw_launcher(self, history_idx, draw_weight=True, save_video=True):

        pos = self.launcher.pos_history[history_idx]
        theta = self.launcher.theta_history[history_idx]

        if draw_weight:
            weight_start_y = self.launcher.r[1]
            pos_weight = self.launcher.pos_weight_history[history_idx]
            rubber_band_height = self.launcher.rubber_band_height_history[history_idx]
            weight_y = weight_start_y + pos_weight - self.launcher.h

            if pos_weight >= rubber_band_height:
                self.rope_vertical.set_data([self.weight_x, self.weight_x], [weight_start_y, weight_y])
                self.rubber_band.set_data([], [])

            else:
                rubber_band_top_y = weight_start_y - self.launcher.h + rubber_band_height
                self.rope_vertical.set_data([self.weight_x, self.weight_x], [weight_start_y, rubber_band_top_y])
                self.rubber_band.set_data([self.weight_x, self.weight_x], [rubber_band_top_y, weight_y])

            self.weight_point.set_data([self.weight_x], [weight_y])

        for i in range(self.launcher.n_joints):
            c_mag = self.launcher.c_mag[i]
            l_mag = self.launcher.l_mag[i]

            unit_vec = np.array([np.cos(theta[i]), np.sin(theta[i])])

            start = pos[i] - c_mag * unit_vec
            end = pos[i] + (l_mag - c_mag) * unit_vec

            self.rod_lines[i].set_data([start[0], end[0]], [[start[1], end[1]]])

        if save_video:
            self.canvas_to_video_frame()

    def animate_weight_initial_fall(self):

        self.draw_launcher(0, draw_weight=False, save_video=False)

        t_fall = np.sqrt(2*self.launcher.h/g_mag)
        t_fall_steps = np.arange(0, t_fall, 1/self.fps)

        weight_start_y = self.launcher.r[1]
        for t_f in t_fall_steps:
            weight_y = weight_start_y - 0.5 * g_mag * t_f ** 2

            self.rope_vertical.set_data([self.weight_x, self.weight_x], [weight_start_y, weight_y])

            self.weight_point.set_data([self.weight_x], [weight_y])

            self.canvas_to_video_frame()

    def canvas_to_video_frame(self):
        self.fig.canvas.draw()
        img = np.array(self.fig.canvas.renderer.buffer_rgba())
        img = img[:, :, :-1]
        self.video.write(img)


def main():

    n_rods = 4
    theta = np.full(n_rods, -np.pi/2)
    l_mag = np.ones(n_rods)
    c_mag = np.full(n_rods, 0.5)
    m = np.ones(n_rods)
    I = np.ones(n_rods)
    r_mag = 0.1
    m_weight = 10.0
    h = 1.0
    k = 50.0
    omega = np.zeros(n_rods)

    dt = 0.001
    T = 5

    launcher = Launcher(theta, l_mag, c_mag, m, I, r_mag, m_weight, h, k, omega)
    launcher.simulate(dt, T)

    print('simulation done, generating video')

    animator = Animator()
    animator.animate(launcher, 'animations/video.mp4')


if __name__ == '__main__':
    main()
