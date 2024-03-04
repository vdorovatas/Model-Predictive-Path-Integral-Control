import numpy as np 
import math

class MPPI_Dynamic():
    def __init__(
            self,
            input_map,
            cones,
            vehicle_width: float = 3.0, # [m]
            vehicle_length: float = 4.0, # [m]
            delta_t: float = 0.05,
            l_f: float = 1.1, # [m]
            l_r: float = 1.4, # [m]
            mass: float = 1000.0, # [kg]
            I_z: float = 1300.0, # [kg*m^2]
            C_f: float = 5000.0 * 2.0, # [N/rad]
            C_r: float = 6000.0 * 2.0, # [N/rad]
            max_steer_abs: float = 0.523, # [rad]
            max_accel_abs: float = 2.000, # [m/s^2]
            ref_path: np.ndarray = np.array([[0.0, 0.0, 0.0, 1.0], [10.0, 0.0, 0.0, 1.0]]),
            horizon_step_T: int = 30,
            number_of_samples_K: int = 1000,
            param_exploration: float = 0.0,
            param_lambda: float = 50.0,
            param_alpha: float = 1.0,
            sigma: np.ndarray = np.array([[0.5, 0.0], [0.0, 0.1]]), 
            stage_cost_weight: np.ndarray = np.array([50.0, 50.0, 1.0, 20.0]), # weight for [x, y, yaw, v]
            terminal_cost_weight: np.ndarray = np.array([50.0, 50.0, 1.0, 20.0]), # weight for [x, y, yaw, v]
            visualize_optimal_traj = True,  # if True, optimal trajectory is visualized
            visualze_sampled_trajs = False, # if True, sampled trajectories are visualized
    ) -> None:
        """initialize mppi controller for path-tracking"""
        # mppi parameters
        self.dim_x = 6 # dimension of system state vector
        self.dim_u = 2 # dimension of control input vector
        self.T = horizon_step_T # prediction horizon
        self.K = number_of_samples_K # number of sample trajectories
        self.param_exploration = param_exploration  # constant parameter of mppi
        self.param_lambda = param_lambda  # constant parameter of mppi
        self.param_alpha = param_alpha # constant parameter of mppi
        self.param_gamma = self.param_lambda * (1.0 - (self.param_alpha))  # constant parameter of mppi
        self.Sigma = sigma # deviation of noise
        self.stage_cost_weight = stage_cost_weight
        self.terminal_cost_weight = terminal_cost_weight
        self.visualize_optimal_traj = visualize_optimal_traj
        self.visualze_sampled_trajs = visualze_sampled_trajs

        # vehicle parameters
        self.l_f = l_f # [m]
        self.l_r = l_r # [m]
        self.wheel_base = l_f + l_r # [m]
        self.mass = mass # [kg]
        self.I_z = I_z # [kg*m^2]
        self.C_f = C_f # [N/rad]
        self.C_r = C_r # [N/rad]
        self.max_steer_abs = max_steer_abs # [rad]
        self.max_accel_abs = max_accel_abs # [m/s^2]
        self.delta_t = delta_t #[s]
        self.ref_path = ref_path
        self.c1 = input_map[0]
        self.c2 = input_map[1]
        self.cones = cones
        self.crash = 0
        self.vehicle_w = vehicle_width #[m]
        self.vehicle_l = vehicle_length #[m]

        # mppi variables
        self.u_prev = np.zeros((self.T, self.dim_u))
        # ref_path info
        self.prev_waypoints_idx = 0

    def calc_control_input(self, observed_x: np.ndarray) -> tuple[float, np.ndarray]:
        """calculate optimal control input"""
        # load privious control input sequence
        u = self.u_prev

        # set initial x value from observation
        x0 = observed_x

        # get the waypoint closest to current vehicle position 
        # self._get_nearest_waypoint(x0[0], x0[1], update_prev_idx=True)
        # if self.prev_waypoints_idx >= self.ref_path.shape[0]-1:
        #     print("[ERROR] Reached the end of the reference path.")
        #     raise IndexError

        # prepare buffer
        S = np.zeros((self.K)) # state cost list

        # sample noise
        epsilon = self._calc_epsilon(self.Sigma, self.K, self.T, self.dim_u) # size is self.K x self.T

        # prepare buffer of sampled control input sequence
        v = np.zeros((self.K, self.T, self.dim_u)) # control input sequence with noise

        # loop for 0 ~ K-1 samples
        # all_x = []
        # all_y = []
        for k in range(self.K):         

            # set initial(t=0) state x i.e. observed state of the vehicle
            x = x0
            # loop for time step t = 1 ~ T
            for t in range(1, self.T+1):

                # get control input with noise
                if k < (1.0-self.param_exploration)*self.K:
                    v[k, t-1] = u[t-1] + epsilon[k, t-1] # sampling for exploitation
                else:
                    v[k, t-1] = epsilon[k, t-1] # sampling for exploration

                # update x
                x = self._F(x, self._g(v[k, t-1]))

                # add stage cost
                S[k] += self._c(x) + self.param_gamma * u[t-1].T @ np.linalg.inv(self.Sigma) @ v[k, t-1]

                #print(x0[1], x[1])
            # add terminal cost
            #S[k] += self._phi(x)
            S[k] += 1000000*self.crash
            self.crash = 0
        
        #     all_x.append(x[0])
        #     all_y.append(x[1])
        # print(max(all_x), x0[0])
        # print(max(all_y), min(all_y), x0[1])
        #print(min(S), max(S))

        # compute information theoretic weights for each sample
        w = self._compute_weights(S)

        # calculate w_k * epsilon_k
        w_epsilon = np.zeros((self.T, self.dim_u))
        for t in range(self.T): # loop for time step t = 0 ~ T-1
            for k in range(self.K):
                w_epsilon[t] += w[k] * epsilon[k, t]

        # apply moving average filter for smoothing input sequence
        w_epsilon = self._moving_average_filter(xx=w_epsilon, window_size=10)

        # update control input sequence
        u += w_epsilon

        # calculate optimal trajectory
        optimal_traj = np.zeros((self.T, self.dim_x))
        if self.visualize_optimal_traj:
            x = x0
            for t in range(self.T):
                x = self._F(x, self._g(u[t-1]))
                optimal_traj[t] = x

        #print(optimal_traj[:,1])
        # calculate sampled trajectories
        sampled_traj_list = np.zeros((self.K, self.T, self.dim_x))
        sorted_idx = np.argsort(S) # sort samples by state cost, 0th is the best sample

        if self.visualze_sampled_trajs:
            for k in sorted_idx:
                x = x0
                for t in range(self.T):
                    x = self._F(x, self._g(v[k, t-1]))

                    sampled_traj_list[k, t] = x

        # update previous control input sequence (shift 1 step to the left)
        self.u_prev[:-1] = u[1:]
        self.u_prev[-1] = u[-1]

        # return optimal control input and input sequence
        return u[0], u, optimal_traj, sampled_traj_list

    def _calc_epsilon(self, sigma: np.ndarray, size_sample: int, size_time_step: int, size_dim_u: int) -> np.ndarray:
        """sample epsilon"""
        # check if sigma row size == sigma col size == size_dim_u and size_dim_u > 0
        if sigma.shape[0] != sigma.shape[1] or sigma.shape[0] != size_dim_u or size_dim_u < 1:
            print("[ERROR] sigma must be a square matrix with the size of size_dim_u.")
            raise ValueError
            

        # sample epsilon
        mu = np.zeros((size_dim_u)) # set average as a zero vector
        epsilon = np.random.multivariate_normal(mu, sigma, (size_sample, size_time_step))
        return epsilon

    def _g(self, v: np.ndarray) -> float:
        """clamp input"""
        # limit control inputs
        v[0] = np.clip(v[0], -self.max_steer_abs, self.max_steer_abs) # limit steering input
        v[1] = np.clip(v[1], -self.max_accel_abs, self.max_accel_abs) # limit acceleraiton input
        return v

    # def _c(self, x_t: np.ndarray) -> float:
    #     """calculate stage cost"""
    #     # parse x_t
    #     x, y, yaw, vx, vy, omega = x_t
    #     yaw = ((yaw + 2.0*np.pi) % (2.0*np.pi)) # normalize theta to [0, 2*pi]

    #     v = vx # np.sqrt(vx**2 + vy**2)
    #     #######
    #     _, c1_x_closer, c1_y_closer = self._get_nearest_cones(x,y)
    #     danger = 0
    #     if (np.sqrt((x-c1_x_closer)**2 + (y-c1_y_closer)**2) < 1.7):
    #         danger = 1
    #     # _, c2_x_closer, c2_y_closer = self._get_nearest_cones(x,y)
    #     # calculate stage cost
    #     _, ref_x, ref_y, ref_yaw, ref_v = self._get_nearest_waypoint(x, y)
    #     stage_cost = self.stage_cost_weight[0]*(x-ref_x)**2 + self.stage_cost_weight[1]*(y-ref_y)**2 + \
    #                  self.stage_cost_weight[2]*(yaw-ref_yaw)**2 + self.stage_cost_weight[3]*(v-ref_v)**2 + \
    #                  1000000*danger
    #     return stage_cost

    def _phi(self, x_T: np.ndarray) -> float:
        """calculate terminal cost"""
        # parse x_T
        x, y, yaw, vx, vy, omega = x_T
        yaw = ((yaw + 2.0*np.pi) % (2.0*np.pi)) # normalize theta to [0, 2*pi]

        v = vx # np.sqrt(vx**2 + vy**2)
        #######
        _, c1_x_closer, c1_y_closer = self._get_nearest_cones(x,y)
        danger = 0
        if (np.sqrt((x-c1_x_closer)**2 + (y-c1_y_closer)**2) < 1.7):
            danger = 1
        # calculate terminal cost
        _, ref_x, ref_y, ref_yaw, ref_v = self._get_nearest_waypoint(x, y)
        terminal_cost = self.terminal_cost_weight[0]*(x-ref_x)**2 + self.terminal_cost_weight[1]*(y-ref_y)**2 + \
                        self.terminal_cost_weight[2]*(yaw-ref_yaw)**2 + self.terminal_cost_weight[3]*(v-ref_v)**2 + \
                        1000000*danger
        return terminal_cost

    def _get_nearest_waypoint(self, x: float, y: float, update_prev_idx: bool = False):
        """search the closest waypoint to the vehicle on the reference path"""

        SEARCH_IDX_LEN = 200 # [points] forward search range
        prev_idx = self.prev_waypoints_idx
        dx = [x - ref_x for ref_x in self.ref_path[prev_idx:(prev_idx + SEARCH_IDX_LEN), 0]]
        dy = [y - ref_y for ref_y in self.ref_path[prev_idx:(prev_idx + SEARCH_IDX_LEN), 1]]
        d = [idx ** 2 + idy ** 2 for (idx, idy) in zip(dx, dy)]
        min_d = min(d)
        nearest_idx = d.index(min_d) + prev_idx

        # get reference values of the nearest waypoint
        ref_x = self.ref_path[nearest_idx,0]
        ref_y = self.ref_path[nearest_idx,1]
        ref_yaw = self.ref_path[nearest_idx,2]
        ref_v = self.ref_path[nearest_idx,3]

        # update nearest waypoint index if necessary
        if update_prev_idx:
            self.prev_waypoints_idx = nearest_idx 

        return nearest_idx, ref_x, ref_y, ref_yaw, ref_v
    
    def _get_nearest_cones(self, x: float, y: float, update_prev_idx: bool = False):
        """search the closest waypoint to the vehicle on the reference path"""

        SEARCH_IDX_LEN = 200 # [points] forward search range
        prev_idx = self.prev_waypoints_idx
        dx = [x - ref_x for ref_x in self.c1[prev_idx:(prev_idx + SEARCH_IDX_LEN), 0]]
        dy = [y - ref_y for ref_y in self.c1[prev_idx:(prev_idx + SEARCH_IDX_LEN), 1]]
        d = [idx ** 2 + idy ** 2 for (idx, idy) in zip(dx, dy)]
        min_d = min(d)
        nearest_idx = d.index(min_d) + prev_idx

        # get reference values of the nearest waypoint
        ref_x = self.c1[nearest_idx,0]
        ref_y = self.c1[nearest_idx,1]

        # update nearest waypoint index if necessary
        if update_prev_idx:
            self.prev_waypoints_idx = nearest_idx 

        return nearest_idx, ref_x, ref_y

    def _F(self, x_t: np.ndarray, v_t: np.ndarray) -> np.ndarray:
        """calculate next state of the vehicle"""
        # get previous state variables
        x, y, yaw, vx, vy, omega = x_t
        steer, accel = v_t

        l_f = self.l_f
        l_r = self.l_r
        m = self.mass
        C_f = self.C_f
        C_r = self.C_r
        I_z = self.I_z
        dt = self.delta_t

        """< CORE OF VEHICLE DYNAMICS >"""
        # calculate tire forces
        F_fy = - C_f * ((vy + l_f * omega) / vx - steer)
        F_ry = - C_r * ((vy - l_r * omega) / vx)

        # update state variables
        beta = vy / vx
        new_x = x + (vx * np.cos(yaw) - vy * np.sin(yaw)) * dt
        new_y = y + (vx * np.sin(yaw) + vy * np.cos(yaw)) * dt
        new_yaw = yaw + omega * dt
        new_vx = vx + (accel * np.cos(beta) - (F_fy * np.sin(steer) / m) + vy * omega) * dt
        new_vy = vy + (accel * np.sin(beta) + F_ry / m + (F_fy * np.cos(steer) / m) - vx * omega) * dt
        new_omega = omega + ((F_fy * l_f * np.cos(steer) - F_ry * l_r) / I_z) * dt
        x_t_plus_1 = np.array([new_x, new_y, new_yaw, new_vx, new_vy, new_omega])
        return x_t_plus_1
        """< CORE OF VEHICLE DYNAMICS >"""

    def _compute_weights(self, S: np.ndarray) -> np.ndarray:
        """compute weights for each sample"""
        # prepare buffer
        w = np.zeros((self.K))

        # calculate rho
        rho = S.min()

        # calculate eta
        eta = 0.0
        for k in range(self.K):
            eta += np.exp( (-1.0/self.param_lambda) * (S[k]-rho) )

        # calculate weight
        for k in range(self.K):
            w[k] = (1.0 / eta) * np.exp( (-1.0/self.param_lambda) * (S[k]-rho) )
        return w

    def _moving_average_filter(self, xx: np.ndarray, window_size: int) -> np.ndarray:
        """apply moving average filter for smoothing input sequence
        Ref. https://zenn.dev/bluepost/articles/1b7b580ab54e95
        """
        b = np.ones(window_size)/window_size
        dim = xx.shape[1]
        xx_mean = np.zeros(xx.shape)

        for d in range(dim):
            xx_mean[:,d] = np.convolve(xx[:,d], b, mode="same")
            n_conv = math.ceil(window_size/2)
            xx_mean[0,d] *= window_size/n_conv
            for i in range(1, n_conv):
                xx_mean[i,d] *= window_size/(i+n_conv)
                xx_mean[-i,d] *= window_size/(i + n_conv - (window_size % 2)) 
        return xx_mean


    #######################################3


    def _get_nearest_cone(self, x: float, y: float, c):

        """find the closest cone to the vehicle"""

        closest_points = self.binary_search_closest(x, c)
        distances = np.sqrt((closest_points[:, 0] - x)**2 + (closest_points[:, 1] - y)**2)
        closest_index = np.argmin(distances)
        return closest_points[closest_index]
    

    def binary_search_closest(self, x_value, sorted_array, num_closest=20):
        index = np.searchsorted(sorted_array[:, 0], x_value, side="left")
        start_index = max(0, index - num_closest // 2)
        end_index = min(len(sorted_array), start_index + num_closest)
        return sorted_array[start_index:end_index]
    
    def _c(self, x_t: np.ndarray) -> float:
        """calculate stage cost"""
        # parse x_t
        x, y, yaw, vx, vy, omega = x_t
        yaw = ((yaw + 2.0*np.pi) % (2.0*np.pi)) # normalize theta to [0, 2*pi]

        ref_v = 7.0
        
        C_speed = (vx-ref_v)**2
        ###
        zeta = - np.arctan(vy / np.abs(vx))
        slip = 0
        if (np.abs(zeta) > 0.75): 
            slip = 1
        C_stab = zeta**2 + 10000*slip
        ####
        # C_control = ...
        ####
        closer_cone1_x, closer_cone1_y = self._get_nearest_cone(x,y, self.c1)
        closer_cone2_x, closer_cone2_y = self._get_nearest_cone(x,y, self.c2)

        distance_from_closer_cone1 = np.sqrt((x-closer_cone1_x)**2 + (y-closer_cone1_y)**2)
        distance_from_closer_cone2 = np.sqrt((x-closer_cone2_x)**2 + (y-closer_cone2_y)**2)
        distance_from_closer_cone =  self._get_nearest_cone(x,y, self.cones) #np.abs(distance_from_closer_cone2 - distance_from_closer_cone1) 
        danger = 0
        _, ref_x, ref_y, ref_yaw, ref_v = self._get_nearest_waypoint(x, y)
        if (distance_from_closer_cone1 < 3 or distance_from_closer_cone2 < 3):
            self.crash = 1
            danger = 0
        C_track =  100000000*danger  + 0.08*self.get_dist(x, y, distance_from_closer_cone)#+ 2*((x-ref_x)**2 + (y-ref_y)**2)
        #######
        stage_cost = 2.5*C_speed + 50*C_stab + 100*C_track
        return stage_cost
    

    def _is_collided(self,  x_t: np.ndarray) -> bool:

        # vehicle shape parameters
        vw, vl = self.vehicle_w, self.vehicle_l
        safety_margin_rate = 1.0
        vw, vl = vw*safety_margin_rate, vl*safety_margin_rate

        # get current states
        x, y, yaw, _ = x_t

        # key points for collision check
        vehicle_shape_x = [-0.5*vl, -0.5*vl, 0.0, +0.5*vl, +0.5*vl, +0.5*vl, 0.0, -0.5*vl, -0.5*vl]
        vehicle_shape_y = [0.0, +0.5*vw, +0.5*vw, +0.5*vw, 0.0, -0.5*vw, -0.5*vw, -0.5*vw, 0.0]
        rotated_vehicle_shape_x, rotated_vehicle_shape_y = \
            self._affine_transform(vehicle_shape_x, vehicle_shape_y, yaw, [x, y]) # make the vehicle be at the center of the figure

        # check if the key points are inside the obstacles
        for obs in self.obstacle_circles: # for each circular obstacles
            obs_x, obs_y, obs_r = obs # [m] x, y, radius
            for p in range(len(rotated_vehicle_shape_x)):
                if (rotated_vehicle_shape_x[p]-obs_x)**2 + (rotated_vehicle_shape_y[p]-obs_y)**2 < obs_r**2:
                    return 1.0 # collided

        return 0.0 # not collided
    

    def get_dist(self, x, y, c):
        cx = c[0]
        cy = c[1]
        dist = np.sqrt((cx - x)**2 + (cy - y)**2)
        if dist > 9.9: return 0
        elif dist > 5 : 
            return 5/dist
        else: return 1