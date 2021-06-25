import numpy as np
from numpy import random
from itertools import combinations
from scipy.special import comb

class AoI_Energy(object):
    def __init__(self, seed=0, user_num=24, beta2=1.0, request_p=0.6, **kwargs):
        User_Num = user_num  # number of users
        Sensor_Num = 8  # number of sensors
        Max_Update_Num = 4  # supported maximum number of updated sensors
        Sensing_Energy_Cost = np.full((Sensor_Num,), 10, dtype=np.float64)  # Es (Sensor_Num,)
        Updating_Energy_Cost = np.full((Sensor_Num,), 10, dtype=np.float64)  # Eu (Sensor_Num,)
        User_Req_Prob_Matrix = np.full((User_Num,), request_p, dtype=np.float64)  # probability of request (User_num,)
        Sensor_Popularity = np.full((Sensor_Num,), 1 / Sensor_Num, dtype=np.float64)  # popularity of request (Sensor_Num,)
        Fail_Prob = np.array([0.025, 0.025, 0.05, 0.05, 0.075, 0.075, 0.1, 0.1], dtype=np.float64)  # failure probability (Sensor_Num,)
        Omega_Pro = np.ones(User_Num, dtype=np.float64) * (1 / User_Num)  # weight for different users
        Dd = 1 # slot of Dd
        Du = 1 # slot of Du
        Beta_1 = 1 # weight for AoI cost
        Beta_2 = beta2 # weight for energy cost
        Para = 10  # parameter of AoI_max

        self.N = User_Num  # number of users
        self.K = Sensor_Num  # number of sensors
        self.M = Max_Update_Num  # supported maximum number of updated sensors

        self.Dd = int(Dd)  # data delivery phase
        self.Du = int(Du)  # data update phase

        self.W = Omega_Pro  # weight for different users
        self.B1 = Beta_1
        self.B2 = Beta_2
        self.AoI_Max = Para * self.K * (self.Dd + self.Du)
        self.Fail_Prob_Pro = 1 - np.power((1 - Fail_Prob), self.Du)  # failure probability
        self.E = np.zeros(self.K)
        self.E = Sensing_Energy_Cost + Updating_Energy_Cost  # Et = Es + Eu, total energy consumption for each sensor
        self.Cost_Upper_Bound = self.B1 * sum(self.W * self.AoI_Max) + self.B2 * sum(self.E[0:self.M]) # max cost
        self.Com_Factor = 1  # compensation factor
        self.C1 = self.Com_Factor * self.Cost_Upper_Bound
        self.User_Req_Prob = np.zeros((self.N, self.K + 1))
        self.User_Req_Prob[:, 1:(self.K + 1)] = User_Req_Prob_Matrix.reshape(-1, 1) * Sensor_Popularity  # probability of users' request
        self.User_Req_Prob[:, 0] = 1 - np.sum(self.User_Req_Prob[:, 1:self.K + 1], axis=1)  # the first column is the probability of no request (self.N, self.K+1)
        self.State_Space = np.zeros((self.N + 1, self.K))  # state, including AoI of ECN and users
        self.Actual_Action_Num = self._possible_action_num(self.K, self.M)
        self.Action_Space = np.zeros((self.Actual_Action_Num, self.K), dtype=np.int64)  # (actual_action_num, K)
        self._action_space_fill()  # fill action space
        self.Action_Row = np.zeros(self.K, dtype=np.int64)  # index in action space of an action

    def _possible_action_num(self, Q, P):
        A = 0
        for i in range(0, P + 1):
            A += comb(Q, i, exact=True)
        return A

    def _action_space_fill(self):
        """
        fill the action space using 0 or 1
        """
        characters = np.arange(self.K)
        indices = []
        for i in np.arange(self.M + 1):
            for item in combinations(characters, i):
                indices.append(item)

        for j in range(self.Actual_Action_Num):
            self.Action_Space[j, indices[j]] = 1

    def _user_request_ind(self):
        # simulation user request behavior
        user_req_ind_ = np.zeros(self.N, dtype=np.int64)
        sensor_seq = np.arange(0, self.K + 1, 1)
        for i in range(self.N):
            user_req_ind_[i] = np.random.choice(sensor_seq, p=self.User_Req_Prob[i, :])
        return user_req_ind_

    def _update_state(self):
        # Calculate Dt
        update_flag = sum(self.Action_Row) != 0
        deliver_flag = sum(self.user_req_ind) != 0

        self.D = 0
        if update_flag:
            self.D += self.Du
        if deliver_flag:
            self.D += self.Dd
        if (not update_flag) and (not deliver_flag):
            self.D = 1

        # Update ECN AoI
        Update_Req = np.where(self.Action_Row == 1)[0]
        self.Update_Result = np.zeros(self.K)
        for i in Update_Req:
            value = np.random.choice([0, 1], p=[self.Fail_Prob_Pro[i], 1 - self.Fail_Prob_Pro[i]])
            self.Update_Result[i] = value

        indices_succ = np.where(self.Update_Result == 1)[0]
        indices_fail = np.where(self.Update_Result == 0)[0]
        self.State_Space[0, indices_succ] = self.D
        self.State_Space[0, indices_fail] = self.State_Space[0, indices_fail] + self.D

        # Update user AoI
        self.State_Space[1:self.N + 1, :] = self.State_Space[1:self.N + 1, :] + self.D
        actual_user_req_ind = np.where(self.user_req_ind != 0)[0]
        state_space_update_indices = (actual_user_req_ind + 1, self.user_req_ind[actual_user_req_ind] - 1)
        self.State_Space[state_space_update_indices] = self.State_Space[0, state_space_update_indices[1]]

        # AoI upper bound
        over_indices = np.where(self.State_Space[:, :] > self.AoI_Max)
        self.State_Space[over_indices] = self.AoI_Max

    def _step(self, action):
        assert ((np.arange(self.Actual_Action_Num) == action).any()) == True
        self.Action_Row = self.Action_Space[action, :]
        self._update_state()  # update cache and user state

        self.AoI_Table = np.zeros((self.N, self.K, self.D+1))
        self.AoI_Table[:,:,0] = self.Last_State[1:self.N+1,:]
        for i in range(self.N):
            for j in range(self.K):
                init_aoi = int(self.AoI_Table[i, j, 0]) # initial AoI value
                self.AoI_Table[i, j, 1:self.D]=np.linspace(init_aoi+1, init_aoi+self.D-1, self.D-1, endpoint=True)
                self.AoI_Table[i, j, -1]=self.State_Space[1+i, j]
        self.AoI_Table = np.minimum(self.AoI_Table, self.AoI_Max)
        sensors = np.sum(self.AoI_Table[:,:,1:], axis=2)/self.D # (N,K)

        # a more efficient way to calculate `sensors`
        # sensors = (self.Last_State[1:]+1.+self.State_Space[1:])/2.
        # sensors = np.minimum(sensors, self.AoI_Max)

        users = np.sum(sensors, axis=1) / self.K  # (N,)
        self.AoI_Cost = int(np.sum(self.W * users * 10000)) / 10000
        self.Energy_Cost = int(sum(self.Action_Row * self.E * 10000)) / 10000
        aoi_cost = self.B1 * self.AoI_Cost
        energy_cost = self.B2 * self.Energy_Cost
        self.Total_Cost = aoi_cost + energy_cost
        self.reward = - self.Total_Cost  # real reward

        self.Last_State = self.State_Space.copy()
        return self.State_Space.copy(), (self.reward, aoi_cost, energy_cost)

    def reset(self):
        # initial state
        self.State_Space = np.zeros((self.N + 1, self.K))
        self.Last_State = self.State_Space.copy()
        return self.State_Space.copy()

    def simulation_user_request(self):
        self.user_req_ind = self._user_request_ind()
        return self.user_req_ind.copy()

    def step(self, action):
        return self._step(action)

if __name__ == '__main__':
    env = AoI_Energy()