from launcher import Launcher
import matplotlib.pyplot as plt
import numpy as np
import cv2


def animate(launcher, fps, video_fname):

    n_steps = launcher.pos_history.shape[0]

    frame_jump = int(1/fps / launcher.dt)
    save_idx = np.array(range(0, n_steps, frame_jump))

    fig, ax = plt.subplots(1, 1)
    lines = [ax.plot([], [], 'ko-')[0] for i in range(launcher.n_joints)]

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

    for idx in save_idx:
        pos = launcher.pos_history[idx]
        theta = launcher.theta_history[idx]

        for i in range(launcher.n_joints):
            c_mag = launcher.c_mag[i]
            l_mag = launcher.l_mag[i]

            unit_vec = np.array([np.cos(theta[i]), np.sin(theta[i])])

            start = pos[i] - c_mag * unit_vec
            end = pos[i] + (l_mag - c_mag) * unit_vec

            lines[i].set_data([ start[0], end[0] ], [ [start[1], end[1]] ])

        fig.canvas.draw()

        img = np.array(fig.canvas.renderer.buffer_rgba())
        img = img[:, :, :-1]
        video.write(img)

    video.release()


def main():

    n_rods = 4
    theta = np.full(n_rods, -np.pi/2)
    l_mag = np.ones(n_rods)
    c_mag = np.full(n_rods, 0.5)
    m = np.ones(n_rods)
    I = np.ones(n_rods)
    r = np.array([0, 0.1, 0])
    omega = np.ones(n_rods)

    dt = 0.001
    T = 5
    fps = 30

    launcher = Launcher(theta, l_mag, c_mag, m, I, r, omega)
    launcher.simulate(dt, T)

    print('simulation done, generating video')

    animate(launcher, fps, 'animations/video.mp4')

    plt.show()


if __name__ == '__main__':
    main()
