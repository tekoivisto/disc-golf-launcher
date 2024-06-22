import numpy as np
import scipy


g = np.array([0, -9.81])
g_mag = np.linalg.norm(g)
# Air
rho = 1.225


def cross_vec(a, b):
    """Faster than np.cross when operating on single vectors instead
    of large arrays of vectors"""
    return a[0]*b[1] - a[1]*b[0]


def cross_arr(a, b):
    """Faster than np.cross when operating on small arrays of vectors"""
    return a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]


class Launcher:

    def __init__(self, params, use_gravity=False):

        self.dt = None
        self.use_gravity = use_gravity

        self.theta = np.deg2rad(np.array(params['theta'], dtype=float))
        self.theta1_initial = self.theta[0]
        self.l_mag = np.array(params['rod_length'], dtype=float)
        self.l_mag = np.append(self.l_mag, params['disc_r']-params['disc_rim_width'])

        self.m = np.array(params['rod_m'], dtype=float)
        self.m = np.append(self.m, params['disc_m'])
        self.mT = self.m.reshape(-1, 1)
        self.n_joints = len(self.theta)
        self.n_rods = self.n_joints-1

        if 'rod_com' in params.keys():
            self.c_mag = np.array(params['rod_com'], dtype=float)
        else:
            self.c_mag = self.l_mag[:-1]/2
        self.c_mag = np.append(self.c_mag, params['disc_r']-params['disc_rim_width'])
        self.c = None
        self.d = None

        if 'rod_I' in params.keys():
            self.I = np.array(params['rod_I'], dtype=float)
        else:
            self.I = 1/12*self.m[:-1]*self.l_mag[:-1]**2
        self.I = np.append(self.I, params['disc_I'])

        if 'omega' in params.keys():
            self.omega = np.array(params['omega'], dtype=float)
        else:
            self.omega = np.zeros(self.n_joints)

        self.m_weight = params['weight_m']
        self.h = params['weight_h']
        self.h_initial = self.h
        self.h_rubber_band = params['rubber_band_h']
        self.h_rubber_band_initial = self.h_rubber_band
        self.k = params['rubber_band_k']
        self.r_mag = params['winch_r']
        self.r = np.array([0.0, self.r_mag])
        self.energy_lost_to_ground = 0.0

        # Weights free fall is not simulated, and simulation start when rubber band starts to stretch
        self.h_weight = self.h_rubber_band
        self.v_weight = -np.sqrt(2*g_mag*(self.h-self.h_rubber_band))

        self.m_disc = params['disc_m']
        self.r_disc = params['disc_r']
        self.I_disc = params['disc_I']
        self.C_drag = params['disc_C_drag']
        self.disc_area = np.pi*self.r_disc**2
        self.energy_lost_to_drag = 0.0

        self.disc_attached = True

        self.pos = np.empty((self.n_joints, 2))
        self.v = np.empty((self.n_joints, 2))
        rod_end = np.zeros(2)
        v_rod_end = np.zeros(2)
        for i in range(self.n_joints):
            theta_i = self.theta[i]
            self.pos[i] = rod_end + self.c_mag[i]*np.array([np.cos(theta_i), np.sin(theta_i)])
            rod_end += self.l_mag[i]*np.array([np.cos(theta_i), np.sin(theta_i)])

            self.v[i] = v_rod_end + self.c_mag[i]*self.omega[i]*np.array([-np.sin(theta_i), np.cos(theta_i)])
            v_rod_end += self.l_mag[i]*self.omega[i]*np.array([-np.sin(theta_i), np.cos(theta_i)])

        self.pos_disc = self.pos[-1]
        self.pos_disc_prev = self.pos[-1]
        self.v_disc = np.zeros(2)
        self.F_drag = np.zeros(2)
        self.F_drag_prev = np.zeros(2)

        self.history = {}
        self.history_disc_released = {}
        
        self.disc_release_timestep = None

    def simulate(self, dt, simulation_length, release_disc=True, use_velocity_verlet=True):

        self.dt = dt
        n_steps = int(simulation_length/dt)

        self.initialize_history(self.history, n_steps+1)
        self.update_history(self.history, 0)

        E_start = self.calculate_total_energy()

        if use_velocity_verlet:
            step_function = self.velocity_verlet_step
        else:
            step_function = self.euler_step

        for i in range(n_steps):
            step_function()
            self.update_history(self.history, i+1)

        E_end = self.calculate_total_energy()

        # print(f'delta E (J):                {E_end-E_start:.3f}')
        # print(f'proportional energy change: {(E_end-E_start)/E_start:.5e}')

        if release_disc:
            self.simulate_disc_release(dt, use_velocity_verlet)

        E_disc, _, _ = self.calculate_disc_energy(self.history)

        return np.amax(E_disc)

    def velocity_verlet_step(self):

        a_weight, a, alpha = self.calc_accelerations()
        self.velocity_verlet_velocity_half_step(a_weight, a, alpha)

        self.h_weight += self.v_weight*self.dt
        self.pos += self.v*self.dt
        self.theta += self.omega*self.dt

        self.check_weight_hitting_ground()
        self.add_energy_lost_to_drag((self.F_drag+self.F_drag_prev)/2)

        a_weight, a, alpha = self.calc_accelerations()
        self.velocity_verlet_velocity_half_step(a_weight, a, alpha)

        self.F_drag_prev = self.F_drag

    def velocity_verlet_velocity_half_step(self, a_weight, a, alpha):

        if self.h_weight > 0 or a_weight > 0:
            self.v_weight = self.v_weight + 0.5*a_weight*self.dt
        else:
            self.v_weight = 0
        self.v = self.v + 0.5*a*self.dt
        self.omega = self.omega + 0.5*alpha*self.dt

    def calc_accelerations(self):

        T = self.calc_T()
        self.F_drag = self.calc_F_drag()
        N = self.solve_normal_forces(T, self.F_drag)

        a_weight = g[1] - T[0]/self.m_weight
        a = self.calc_a_trans(N, T, self.F_drag)
        alpha = self.calc_alpha(N, T, self.F_drag)

        return a_weight, a, alpha

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

        return np.array([-T_mag, 0.0])

    def calc_F_drag(self):

        if not self.disc_attached:
            return np.zeros(2)
        
        self.v_disc = self.v[-1]
        v_mag = np.linalg.norm(self.v_disc)

        if v_mag == 0:
            return np.zeros(2)

        v_hat = self.v_disc/v_mag

        F_drag_mag = 0.5*rho*self.disc_area*self.C_drag*np.sum(self.v_disc**2)

        return -v_hat*F_drag_mag

    def calc_a_trans(self, N, T, F):

        F_tot = np.zeros((self.n_joints, 2))
        F_tot += N
        F_tot[:-1] -= N[1:]
        F_tot[0] += T
        F_tot[-1] += F

        if self.use_gravity:
            F_tot += self.mT*g

        return F_tot / self.mT

    def calc_alpha(self, N, T, F):

        tau = cross_arr(self.c, N)
        tau[:-1] -= cross_arr(self.d[:-1], N[1:])

        tau[-1] += cross_vec(self.d[-1], F)
        tau[0] += cross_vec(self.c[0]+self.r, T)

        return tau / self.I

    def euler_step(self):

        self.check_weight_hitting_ground()

        a_weight, a, alpha = self.calc_accelerations()

        self.euler_update(a_weight, a, alpha)

        self.add_energy_lost_to_drag(self.F_drag)

    def euler_update(self, a_weight, a_trans, alpha):

        self.h_weight += self.v_weight*self.dt
        if self.h_weight > 0 or a_weight > 0:
            self.v_weight += a_weight*self.dt

        self.pos += self.v*self.dt
        self.v += a_trans*self.dt

        self.theta += self.omega*self.dt
        self.omega += alpha*self.dt

    def add_energy_lost_to_drag(self, F_drag):

        if not self.disc_attached:
            return

        self.pos_disc = self.pos[-1]

        dx = self.pos_disc-self.pos_disc_prev
        dW = np.dot(F_drag, dx)

        self.energy_lost_to_drag -= dW

        self.pos_disc_prev = np.copy(self.pos_disc)

    def solve_normal_forces(self, T, F):

        unit_vecs = np.array([np.cos(self.theta), np.sin(self.theta)])
        self.c = - (self.c_mag*unit_vecs).T
        self.d = ((self.l_mag - self.c_mag) * unit_vecs).T

        cx = self.c[:, 0]
        cy = self.c[:, 1]
        dx = self.d[:, 0]
        dy = self.d[:, 1]

        px = 1/self.m[:-1] + cy[:-1]*dy[:-1]/self.I[:-1]
        py = 1/self.m[:-1] + cx[:-1]*dx[:-1]/self.I[:-1]

        qx = - cx[:-1]*dy[:-1]/self.I[:-1]
        qy = - cy[:-1]*dx[:-1]/self.I[:-1]

        rx = np.empty(self.n_joints)
        ry = np.empty(self.n_joints)
        rx[1:] = - 1/self.m[1:] - 1/self.m[:-1] - cy[1:]**2/self.I[1:] - dy[:-1]**2/self.I[:-1]
        ry[1:] = - 1/self.m[1:] - 1/self.m[:-1] - cx[1:]**2/self.I[1:] - dx[:-1]**2/self.I[:-1]
        rx[0] = - 1/self.m[0] - cy[0]**2/self.I[0]
        ry[0] = - 1/self.m[0] - cx[0]**2/self.I[0]

        s = np.empty(self.n_joints)
        s[1:] = cx[1:]*cy[1:]/self.I[1:] + dx[:-1]*dy[:-1]/self.I[:-1]
        s[0] = cx[0]*cy[0]/self.I[0]

        tx = 1/self.m + dy*cy/self.I
        ty = 1/self.m + dx*cx/self.I

        ux = - dx*cy/self.I
        uy = - dy*cx/self.I

        vx = np.empty(self.n_joints)
        vy = np.empty(self.n_joints)
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
        v = np.empty((2*self.n_joints, 1))
        v[::2, 0] = vx
        v[1::2, 0] = vy

        # !!!
        N_sol = scipy.linalg.solve_banded((3, 3), A_band, v, overwrite_ab=True, overwrite_b=True, check_finite=False)

        N = np.empty((self.n_joints, 2))
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

        E_tot = self.energy_lost_to_ground + self.energy_lost_to_drag + E_trans_weight + E_pot_weight + E_spring + E_trans_rods + E_rot_rods

        return E_tot

    def calculate_disc_energy(self, history):
        v_mag = np.linalg.norm(history['v disc'], axis=1)
        E_trans = 0.5 * self.m_disc * v_mag ** 2
        E_rot = 0.5 * self.I_disc * history['omega disc'] ** 2
        E_tot = E_trans + E_rot

        return E_tot, E_trans, E_rot

    def initialize_history(self, history, n_steps):

        history['h weight'] = np.empty(n_steps)
        history['v weight'] = np.empty(n_steps)
        history['h rubber band'] = np.empty(n_steps)
        history['pos'] = np.empty((n_steps, self.n_joints, 2))
        history['v'] = np.empty((n_steps, self.n_joints, 2))
        history['theta'] = np.empty((n_steps, self.n_joints))
        history['omega'] = np.empty((n_steps, self.n_joints))
        if self.disc_attached:
            history['pos disc'] = np.empty((n_steps, 2))
            history['v disc'] = np.empty((n_steps, 2))
            history['omega disc'] = np.empty((n_steps))

    def update_history(self, history, idx):

        history['h weight'][idx] = self.h_weight
        history['v weight'][idx] = self.v_weight
        history['h rubber band'][idx] = self.h_rubber_band
        history['pos'][idx] = self.pos
        history['v'][idx] = self.v
        history['theta'][idx] = self.theta
        history['omega'][idx] = self.omega
        if self.disc_attached:
            history['pos disc'][idx] = self.pos_disc
            history['v disc'][idx] = self.v_disc
            history['omega disc'][idx] = self.omega[-1]

    def simulate_disc_release(self, dt, use_velocity_verlet=True):
        
        self.disc_release_timestep = self.reset_launch_state()

        self.detach_disc()

        n_steps = len(self.history['v disc']) - self.disc_release_timestep
        self.initialize_history(self.history_disc_released, n_steps)
        
        if use_velocity_verlet:
            step_function = self.velocity_verlet_step
        else:
            step_function = self.euler_step
            
        for i in range(n_steps):
            step_function()
            self.update_history(self.history_disc_released, i)

        self.set_released_disc_history(n_steps, dt)

    def reset_launch_state(self):

        E_tot, E_trans, E_rot = self.calculate_disc_energy(self.history)
        self.disc_release_timestep = np.argmax(E_tot)

        self.h_weight = self.history['h weight'][self.disc_release_timestep]
        self.v_weight = self.history['v weight'][self.disc_release_timestep]
        self.h_rubber_band = self.history['h rubber band'][self.disc_release_timestep]
        self.pos = self.history['pos'][self.disc_release_timestep]
        self.v = self.history['v'][self.disc_release_timestep]
        self.theta = self.history['theta'][self.disc_release_timestep]
        self.omega = self.history['omega'][self.disc_release_timestep]
        
        return self.disc_release_timestep
    
    def detach_disc(self):

        self.disc_attached = False

        self.theta = self.theta[:-1]
        self.l_mag = self.l_mag[:-1]
        self.c_mag = self.c_mag[:-1]
        self.m = self.m[:-1]
        self.mT = self.mT[:-1]
        self.I = self.I[:-1]
        self.pos = self.pos[:-1]
        self.v = self.v[:-1]
        self.omega = self.omega[:-1]
        self.n_joints -= 1

    def set_released_disc_history(self, n_steps, dt):

        v_disc = self.history['v disc'][self.disc_release_timestep]
        self.history_disc_released['v disc'] = np.full((n_steps, 2), v_disc)
        self.history_disc_released['omega disc'] = np.full(n_steps, self.history['omega disc'][self.disc_release_timestep])

        t = np.arange(1, n_steps+1)*dt
        start_pos = self.history['pos disc'][self.disc_release_timestep]
        self.history_disc_released['pos disc'] = start_pos + v_disc*t.reshape((-1, 1))
