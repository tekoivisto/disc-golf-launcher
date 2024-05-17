import numpy as np
import scipy


g = np.array([0, -9.81])
g_mag = np.linalg.norm(g)


class Launcher:

    def __init__(self, theta, l_mag, c_mag, m, I, r_mag, m_weight, h, k, omega=None, use_gravity=False):

        self.dt = None
        self.use_gravity = use_gravity

        self.n_joints = len(theta)
        self.theta = theta
        self.theta1_initial = self.theta[0]
        self.l_mag = l_mag
        self.c_mag = c_mag
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
        self.k = k

        self.pos_weight = 0
        self.v_weight = -np.sqrt(2*self.h*g_mag)
        self.rubber_band_height = 0

        self.pos_history = None
        self.theta_history = None
        self.pos_weight_history = None
        self.rubber_band_height_history = None

    def simulate(self, dt, simulation_length):

        E_start = 0.5*(np.sum(self.m*np.linalg.norm(self.v, axis=1)**2 + self.I*self.omega**2)
                       + self.m_weight*self.v_weight**2)

        self.dt = dt
        n_steps = int(simulation_length/dt)

        self.pos_history = np.empty((n_steps+1, self.n_joints, 2))
        self.theta_history = np.empty((n_steps+1, self.n_joints))
        self.pos_weight_history = np.empty(n_steps+1)
        self.rubber_band_height_history = np.empty(n_steps+1)

        self.pos_history[0] = self.pos
        self.theta_history[0] = self.theta
        self.pos_weight_history[0] = self.pos_weight
        self.rubber_band_height_history[0] = self.rubber_band_height

        for i in range(n_steps):
            self.step()

            self.pos_history[i+1] = self.pos
            self.theta_history[i+1] = self.theta
            self.pos_weight_history[i+1] = self.pos_weight
            self.rubber_band_height_history[i+1] = self.rubber_band_height

        if self.pos_weight >= self.rubber_band_height:
            E_spring = 0
        else:
            E_spring = 0.5*self.k*(self.rubber_band_height-self.pos_weight)**2

        E_weight_pot = self.m_weight*g_mag*self.pos_weight
        E_end = 0.5*(np.sum(self.m*np.linalg.norm(self.v, axis=1)**2 + self.I*self.omega**2)
                     + self.m_weight*self.v_weight**2) + E_spring + E_weight_pot

        print('delta E')
        print(E_end-E_start)
        print((E_end-E_start)/E_start)

        return self.pos_history, self.theta_history

    def step(self):

        self.rubber_band_height = -self.r_mag*(self.theta[0]-self.theta1_initial)
        if self.pos_weight >= self.rubber_band_height:
            T_mag = 0
        else:
            T_mag = self.k*(self.rubber_band_height-self.pos_weight)
        T = np.array([-T_mag, 0])

        F = np.zeros(2)
        N = self.solve_normal_forces(T, F)

        F_tot = np.copy(N)
        F_tot[:-1] -= N[1:]
        F_tot[0] += T
        F_tot[-1] += F

        if self.use_gravity:
            F_tot += self.m.reshape(-1, 1)*g

        a = F_tot / self.m.reshape(-1, 1)

        self.pos += self.v*self.dt
        self.v += a*self.dt

        a_weight = g[1] + T_mag/self.m_weight
        self.pos_weight += self.v_weight*self.dt
        self.v_weight += a_weight*self.dt

        l = (self.l_mag * np.array([np.cos(self.theta), np.sin(self.theta), np.zeros(self.n_joints)]))
        c = -self.c_mag/self.l_mag * l
        d = (self.l_mag - self.c_mag) / self.l_mag * l
        c = c.T
        d = d.T

        N = np.hstack((N, np.zeros((self.n_joints, 1))))

        tau = np.cross(c, N)[:, -1]
        tau[:-1] += - np.cross(d[:-1], N[1:])[:, -1]

        tau[-1] += np.cross(d[-1], np.hstack((F, [0])))[-1]
        tau[0] += np.cross(c[0]+self.r, np.hstack((T, [0])))[-1]

        alpha = tau / self.I

        self.theta += self.omega*self.dt
        self.omega += alpha*self.dt

    def solve_normal_forces(self, T, F):

        l = self.l_mag * np.array([np.cos(self.theta), np.sin(self.theta)])
        c = - self.c_mag / self.l_mag * l
        d = (self.l_mag - self.c_mag)/self.l_mag * l
        cx = c[0]
        cy = c[1]
        dx = d[0]
        dy = d[1]

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
