import numpy as np
import scipy


g = np.array([0, -9.81])
g_mag = np.linalg.norm(g)


class Launcher:

    def __init__(self, theta, l_mag, c_mag, m, I, r_mag, m_weight, h, h_rubber_band, k, omega=None, use_gravity=False):

        self.dt = None
        self.use_gravity = use_gravity

        self.n_joints = len(theta)
        self.theta = theta
        self.theta1_initial = self.theta[0]
        self.l_mag = l_mag
        self.c_mag = c_mag
        self.c = None
        self.d = None
        self.m = m
        self.I = I

        if omega is None:
            self.omega = np.zeros(self.n_joints)
        else:
            self.omega = omega

        self.pos = np.empty((self.n_joints, 2))
        self.v = np.empty((self.n_joints, 2))
        rod_end = np.zeros(2)
        v_rod_end = np.zeros(2)
        for i in range(self.n_joints):
            theta = self.theta[i]
            self.pos[i] = rod_end + self.c_mag[i]*np.array([np.cos(theta), np.sin(theta)])
            rod_end += self.l_mag[i]*np.array([np.cos(theta), np.sin(theta)])

            self.v[i] = v_rod_end + self.c_mag[i]*self.omega[i]*np.array([-np.sin(theta), np.cos(theta)])
            v_rod_end += self.l_mag[i]*self.omega[i]*np.array([-np.sin(theta), np.cos(theta)])

        self.r_mag = r_mag
        self.r = np.array([0.0, self.r_mag, 0.0])
        self.m_weight = m_weight
        self.h = h
        self.h_rubber_band = h_rubber_band
        self.h_rubber_band_initial = self.h_rubber_band
        self.k = k

        self.h_weight = h_rubber_band
        self.v_weight = -np.sqrt(2*g_mag*(self.h-self.h_rubber_band))
        self.energy_lost_to_ground = 0.0

        self.pos_history = None
        self.theta_history = None
        self.h_weight_history = None
        self.h_rubber_band_history = None

    def simulate(self, dt, simulation_length):

        self.dt = dt
        n_steps = int(simulation_length/dt)

        self.initialize_history(n_steps)

        E_start = self.calculate_total_energy()

        for i in range(n_steps):

            self.step()

            self.update_history(i)

        E_end = self.calculate_total_energy()

        print(f'delta E (J):                {E_end-E_start:.3f}')
        print(f'fractional energy increase: {(E_end-E_start)/E_start:.3e}')

        return self.pos_history, self.theta_history

    def step(self):

        self.check_weight_hitting_ground()

        T = self.calc_T()
        F = np.zeros(2)
        N = self.solve_normal_forces(T, F)

        a_weight = g[1] - T[0]/self.m_weight
        a = self.calc_a_trans(N, T, F)
        alpha = self.calc_alpha(N, T, F)

        self.euler_step(a_weight, a, alpha)

    def check_weight_hitting_ground(self):

        if self.h_weight < 0 and self.v_weight < 0:
            self.energy_lost_to_ground += 0.5*self.m_weight*self.v_weight**2
            self.v_weight = 0

    def calc_T(self):

        self.h_rubber_band = self.h_rubber_band_initial - self.r_mag*(self.theta[0]-self.theta1_initial)

        if self.h_weight >= self.h_rubber_band:
            T_mag = 0
        else:
            T_mag = self.k*(self.h_rubber_band-self.h_weight)

        return np.array([-T_mag, 0])

    def calc_a_trans(self, N, T, F):

        F_tot = np.zeros((self.n_joints, 2))
        F_tot += N
        F_tot[:-1] -= N[1:]
        F_tot[0] += T
        F_tot[-1] += F

        if self.use_gravity:
            F_tot += self.m.reshape(-1, 1)*g

        return F_tot / self.m.reshape(-1, 1)

    def calc_alpha(self, N, T, F):

        c = self.c.T
        d = self.d.T

        N = np.hstack((N, np.zeros((self.n_joints, 1))))

        tau = np.cross(c, N)[:, -1]
        tau[:-1] -= np.cross(d[:-1], N[1:])[:, -1]

        tau[-1] += np.cross(d[-1], np.hstack((F, [0])))[-1]
        tau[0] += np.cross(c[0]+self.r, np.hstack((T, [0])))[-1]

        return tau / self.I

    def euler_step(self, a_weight, a_trans, alpha):

        self.h_weight += self.v_weight*self.dt
        if self.h_weight > 0 or a_weight > 0:
            self.v_weight += a_weight*self.dt

        self.pos += self.v*self.dt
        self.v += a_trans*self.dt

        self.theta += self.omega*self.dt
        self.omega += alpha*self.dt

    def solve_normal_forces(self, T, F):

        unit_vecs = np.array([np.cos(self.theta), np.sin(self.theta), np.zeros(self.n_joints)])
        self.c = - self.c_mag*unit_vecs
        self.d = (self.l_mag - self.c_mag) * unit_vecs
        cx = self.c[0]
        cy = self.c[1]
        dx = self.d[0]
        dy = self.d[1]

        px = 1/self.m[:-1] + cy[:-1]*dy[:-1]/self.I[:-1]
        py = 1/self.m[:-1] + cx[:-1]*dx[:-1]/self.I[:-1]

        qx = - cx[:-1]*dy[:-1]/self.I[:-1]
        qy = - cy[:-1]*dx[:-1]/self.I[:-1]

        rx = np.zeros(self.n_joints)
        ry = np.zeros(self.n_joints)
        rx[1:] = - 1/self.m[1:] - 1/self.m[:-1] - cy[1:]**2/self.I[1:] - dy[:-1]**2/self.I[:-1]
        ry[1:] = - 1/self.m[1:] - 1/self.m[:-1] - cx[1:]**2/self.I[1:] - dx[:-1]**2/self.I[:-1]
        rx[0] = - 1/self.m[0] - cy[0]**2/self.I[0]
        ry[0] = - 1/self.m[0] - cx[0]**2/self.I[0]

        s = np.zeros(self.n_joints)
        s[1:] = cx[1:]*cy[1:]/self.I[1:] + dx[:-1]*dy[:-1]/self.I[:-1]
        s[0] = cx[0]*cy[0]/self.I[0]

        tx = 1/self.m + dy*cy/self.I
        ty = 1/self.m + dx*cx/self.I

        ux = - dx*cy/self.I
        uy = - dy*cx/self.I

        vx = np.zeros(self.n_joints)
        vy = np.zeros(self.n_joints)
        vx[1:] = self.omega[:-1]**2*dx[:-1] - self.omega[1:]**2*cx[1:]
        vy[1:] = self.omega[:-1]**2*dy[:-1] - self.omega[1:]**2*cy[1:]
        vx[-1] += tx[-1]*F[0] + ux[-1]*F[1]
        vy[-1] += ty[-1]*F[1] + uy[-1]*F[0]
        vx[0] = - self.omega[0]**2*cx[0] + T[0]/self.m[0] + (cy[0]+self.r[1])*cy[0]*T[0]/self.I[0] - (cx[0]+self.r[0])*cy[0]*T[1]/self.I[0]
        vy[0] = - self.omega[0]**2*cy[0] + T[1]/self.m[0] + (cx[0]+self.r[0])*cx[0]*T[1]/self.I[0] - (cy[0]+self.r[1])*cx[0]*T[0]/self.I[0]
        if self.n_joints > 1:
            vx[1] += - T[0]/self.m[0] - (cy[0]+self.r[1])*dy[0]*T[0]/self.I[0] + (cx[0]+self.r[0])*dy[0]*T[1]/self.I[0]
            vy[1] += - T[1]/self.m[0] - (cx[0]+self.r[0])*dx[0]*T[1]/self.I[0] + (cy[0]+self.r[1])*dx[0]*T[0]/self.I[0]
        if self.use_gravity:
            vx[0] += g[0]
            vy[0] += g[1]

        A_band = np.zeros((7, 2*self.n_joints))

        A_band[0, 3::2] = ux[:-1]
        A_band[1, 2::2] = tx[:-1]
        A_band[1, 3::2] = ty[:-1]
        A_band[2, 1::2] = s
        A_band[2, 2::2] = uy[:-1]
        A_band[3, 0::2] = rx
        A_band[3, 1::2] = ry
        A_band[4, 0:-1:2] = s
        A_band[4, 1:-1:2] = qx
        A_band[5, 0:-2:2] = px
        A_band[5, 1:-2:2] = py
        A_band[6, 0:-3:2] = qy
        v = np.zeros((2*self.n_joints, 1))
        v[::2, 0] = vx
        v[1::2, 0] = vy

        # !!!
        N_sol = scipy.linalg.solve_banded((3, 3), A_band, v)

        N = np.zeros((self.n_joints, 2))
        N[:, 0] = N_sol[::2, 0]
        N[:, 1] = N_sol[1::2, 0]

        return N

    def calculate_total_energy(self):

        E_trans_weight = 0.5*self.m_weight*self.v_weight**2
        E_pot_weight = self.m_weight * g_mag * self.h_weight
        if self.h_weight >= self.h_rubber_band:
            E_spring = 0
        else:
            E_spring = 0.5*self.k*(self.h_rubber_band-self.h_weight)**2

        E_trans_rods = 0.5*np.sum(self.m*np.sum(self.v**2, axis=1), axis=0)
        E_rot_rods = 0.5*np.sum(self.I*self.omega**2)

        E_tot = self.energy_lost_to_ground + E_trans_weight + E_pot_weight + E_spring + E_trans_rods + E_rot_rods

        return E_tot

    def initialize_history(self, n_steps):

        self.pos_history = np.empty((n_steps+1, self.n_joints, 2))
        self.theta_history = np.empty((n_steps+1, self.n_joints))
        self.h_weight_history = np.empty(n_steps+1)
        self.h_rubber_band_history = np.empty(n_steps+1)

        self.pos_history[0] = self.pos
        self.theta_history[0] = self.theta
        self.h_weight_history[0] = self.h_weight
        self.h_rubber_band_history[0] = self.h_rubber_band

    def update_history(self, idx):

        idx += 1
        self.pos_history[idx] = self.pos
        self.theta_history[idx] = self.theta
        self.h_weight_history[idx] = self.h_weight
        self.h_rubber_band_history[idx] = self.h_rubber_band
