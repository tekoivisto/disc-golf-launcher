from launcher import Launcher
from launcher import g_mag
import matplotlib.pyplot as plt
import numpy as np
import cv2


def draw_launcher(launcher, history_idx, fig, lines, weight_point, rope_vertical,
                  rubber_band, weight_x, draw_weight=True):
    pos = launcher.pos_history[history_idx]
    theta = launcher.theta_history[history_idx]

    if draw_weight:
        weight_start_y = launcher.r[1]
        pos_weight = launcher.pos_weight_history[history_idx]
        rubber_band_height = launcher.rubber_band_height_history[history_idx]
        weight_y = weight_start_y + pos_weight - launcher.h

        if pos_weight >= rubber_band_height:
            rope_vertical.set_data([weight_x, weight_x], [weight_start_y, weight_y])
            rubber_band.set_data([], [])

        else:
            rubber_band_top_y = weight_start_y-launcher.h+rubber_band_height
            rope_vertical.set_data([weight_x, weight_x], [weight_start_y, rubber_band_top_y])
            rubber_band.set_data([weight_x, weight_x], [rubber_band_top_y, weight_y])

        weight_point.set_data([weight_x], [weight_y])

    for i in range(launcher.n_joints):
        c_mag = launcher.c_mag[i]
        l_mag = launcher.l_mag[i]

        unit_vec = np.array([np.cos(theta[i]), np.sin(theta[i])])

        start = pos[i] - c_mag * unit_vec
        end = pos[i] + (l_mag - c_mag) * unit_vec

        lines[i].set_data([start[0], end[0]], [[start[1], end[1]]])

    fig.canvas.draw()

    img = np.array(fig.canvas.renderer.buffer_rgba())
    img = img[:, :, :-1]

    return img


def draw_weight_fall(launcher, weight_point, rope_vertical, weight_x, fig, video, fps):

    t_fall = np.sqrt(2*launcher.h/g_mag)
    t_fall_steps = np.arange(0, t_fall, 1/fps)

    weight_start_y = launcher.r[1]
    for t_f in t_fall_steps:
        weight_y = weight_start_y - 0.5 * g_mag * t_f ** 2

        rope_vertical.set_data([weight_x, weight_x], [weight_start_y, weight_y])

        weight_point.set_data([weight_x], [weight_y])

        fig.canvas.draw()

        img = np.array(fig.canvas.renderer.buffer_rgba())
        img = img[:, :, :-1]
        video.write(img)


def animate(launcher, fps, video_fname):

    n_steps = launcher.pos_history.shape[0]

    frame_jump = int(1/fps / launcher.dt)
    save_idx = np.array(range(0, n_steps, frame_jump))

    weight_x = -3

    fig, ax = plt.subplots(1, 1)
    lines = [ax.plot([], [], 'ko-')[0] for i in range(launcher.n_joints)]
    weight_point = ax.plot([], [], 'ko')[0]
    rope_color = 'tab:brown'
    rope_horizontal = ax.plot([weight_x, launcher.r[0]], [launcher.r[1], launcher.r[1]], '-', color=rope_color)[0]
    rope_vertical = ax.plot([], [], '-', color=rope_color)[0]
    rubber_band = ax.plot([], [], '-', color='tab:green')[0]

    lim = 6
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_xlim([-lim, lim])
    ax.set_ylim([-lim, lim])
    ax.set_aspect('equal')

    fig.canvas.draw()

    img = fig.canvas.renderer.buffer_rgba()
    height = img.shape[0]
    width = img.shape[1]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_fname, fourcc, fps, (width, height))

    draw_launcher(launcher, 0, fig, lines, weight_point, rope_vertical, rubber_band, weight_x, False)
    draw_weight_fall(launcher, weight_point, rope_vertical, weight_x, fig, video, fps)

    for idx in save_idx:
        img = draw_launcher(launcher, idx, fig, lines, weight_point, rope_vertical, rubber_band, weight_x)
        video.write(img)

    video.release()


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
    fps = 30

    launcher = Launcher(theta, l_mag, c_mag, m, I, r_mag, m_weight, h, k, omega)
    launcher.simulate(dt, T)

    print('simulation done, generating video')

    animate(launcher, fps, 'animations/video.mp4')

    plt.show()


if __name__ == '__main__':
    main()
