from AoI_Energy import AoI_Energy
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import time
import argparse

#################### params ###########################
parser = argparse.ArgumentParser(description='Hyper_params')
parser.add_argument('--Info', default='', type=str)  # information added to log dir name

parser.add_argument('--Seed', default=0, type=int)
parser.add_argument('--Units', default=256, type=int)  # hidden units num of NN
parser.add_argument('--Lr', default=0.0005, type=float)  # learning rate
parser.add_argument('--Lr_Decay', default=1e-5, type=float)
parser.add_argument('--R_Beta', default=0.0005, type=int)  # learning rate for average reward
parser.add_argument('--Max_Epsilon', default=1.0, type=float)
parser.add_argument('--Min_Epsilon', default=0.01, type=float)
parser.add_argument('--Epsilon_Decay', default=1.0, type=float)
parser.add_argument('--Batch_Size', default=64, type=int)
parser.add_argument('--Memory_Size', default=200000, type=int) # buffer size
parser.add_argument('--Start_Size', default=50000, type=int)  # step to begin train
parser.add_argument('--Update_Lazy_Step', default=2000, type=int)  # frequency of target update

parser.add_argument('--Evaluate_Interval', default=2000, type=int) # how often to evaluate (in steps)
parser.add_argument('--Test_Epsilon', default=0.05, type=float) # adopted epsilon during evaluation
parser.add_argument('--Points', default=120, type=int)  # total evaluation times
parser.add_argument('--Test_Step', default=10000, type=int) # step number of one evaluation

parser.add_argument('--Alg', default='dqn_R', type=str)
parser.add_argument('--Gpu_Id', default="-1", type=str) # -1 means CPU

parser.add_argument('--User', default=24, type=int)  # user number
parser.add_argument('--Beta2', default=1.0, type=float)  # B2
parser.add_argument('--Request_P', default=0.6, type=float)  # request prob
parser.add_argument('--Gamma', default=0.95, type=float)  # request prob

args = parser.parse_args()

#################### seed ###########################
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = args.Gpu_Id
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
tf.random.set_seed(args.Seed)
np.random.seed(args.Seed)

#################### log ###########################
# create log file
time_str = time.strftime("%m-%d_%H-%M", time.localtime())
alg = args.Alg
log_dir_name = time_str + '_' + alg + args.Info + '_n' + \
               str(args.User) + '_seed' + str(args.Seed)
fw = tf.summary.create_file_writer(log_dir_name)  # log file witer

# create dir to save model
if not os.path.exists(log_dir_name + '/models'):
    os.makedirs(log_dir_name + '/models')

# save params to a .txt file
prams_file = open(log_dir_name + '/prams_table.txt', 'w')
prams_file.writelines(f'{i:50} {v}\n' for i, v in args.__dict__.items())
prams_file.close()

###################### env ###############################
env = AoI_Energy(user_num=args.User, beta2=args.Beta2,seed=args.Seed, request_p=args.Request_P)
Action_Num = env.Actual_Action_Num
Initial_R = - env.C1

###################### others ###############################
Optimizer = tf.optimizers.Adam(args.Lr, decay=args.Lr_Decay)
W_Initializer = tf.initializers.he_normal(args.Seed)  # NN initializer
Epsilon_Decay_Rate = (args.Min_Epsilon - args.Max_Epsilon) / (args.Memory_Size) * args.Epsilon_Decay # factor of decay
TENSOR_FLOAT_TYPE = tf.dtypes.float32
TENSOR_INT_TYPE = tf.dtypes.int32

class ReplayBuffer:
    def __init__(self, size):
        self.cap = size
        buffer_s_dim = (size, env.N + 1, env.K)

        self.s_buffer = np.empty(buffer_s_dim, dtype=np.float32)
        self.a_buffer = np.random.randint(0, Action_Num, (self.cap, 1), dtype=np.int32)
        self.r_buffer = np.empty((self.cap, 1), dtype=np.float32)
        self.next_s_buffer = np.empty(buffer_s_dim, dtype=np.float32)

        self.cap_index = 0
        self.size = 0

    def store(self, step):
        s, a, r, next_s = step
        self.s_buffer[self.cap_index] = s
        self.a_buffer[self.cap_index][0] = a
        self.r_buffer[self.cap_index][0] = r
        self.next_s_buffer[self.cap_index] = next_s

        self.cap_index = (self.cap_index + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, batch_size)

        batch_s = self.s_buffer[idx]
        batch_a = self.a_buffer[idx]
        batch_r = self.r_buffer[idx]
        batch_next_s = self.next_s_buffer[idx]

        return batch_s, batch_a, batch_r, batch_next_s

    def size(self):
        return self.size

class dqn_agent:
    def __init__(self, max_epsilon, batch_size, memory_size):

        def build_dueling_net():
            inputs = keras.Input(shape=(env.N + 1, env.K))
            x = keras.layers.Flatten()(inputs)

            # v(s)
            v_dense = keras.layers.Dense(args.Units / 2, activation='relu', kernel_initializer=W_Initializer)(x)
            v_dense = keras.layers.Dense(args.Units / 2, activation='relu', kernel_initializer=W_Initializer)(v_dense)
            v_out = keras.layers.Dense(1, kernel_initializer=W_Initializer)(v_dense)

            # advantages
            adv_dense = keras.layers.Dense(args.Units / 2, activation='relu', kernel_initializer=W_Initializer)(x)
            adv_dense = keras.layers.Dense(args.Units / 2, activation='relu', kernel_initializer=W_Initializer)(adv_dense)
            adv_out = keras.layers.Dense(Action_Num, kernel_initializer=W_Initializer)(adv_dense)
            adv_normal = keras.layers.Lambda(lambda x: x - tf.reduce_mean(x))(adv_out)

            # q
            outputs = keras.layers.add([v_out, adv_normal])
            model = keras.Model(inputs=inputs, outputs=outputs)
            return model

        def build_net():
            inputs = keras.Input(shape=(env.N + 1, env.K))
            x = keras.layers.Flatten()(inputs)
            x = keras.layers.Dense(args.Units, activation='relu', kernel_initializer=W_Initializer)(x)
            x = keras.layers.Dense(args.Units, activation='relu', kernel_initializer=W_Initializer)(x)
            outputs = keras.layers.Dense(Action_Num, kernel_initializer=W_Initializer)(x)
            model = keras.Model(inputs=inputs, outputs=outputs)
            return model

        if 'due' in alg:  # dueling
            self.active_qnet = build_dueling_net()
            self.lazy_qnet = build_dueling_net() # target q
            print("dueling net")
        elif 'dqn' in alg:  # dqn
            self.active_qnet = build_net()
            self.lazy_qnet = build_net()
            print("dqn net")
        else:
            raise NotImplementedError("alg not implemented")

        self.active_qnet.compile(optimizer=Optimizer, loss='mse')
        self.epsilon = max_epsilon
        self.batch_size = batch_size
        self.buffer = ReplayBuffer(memory_size)
        self.alg = alg
        self.R = Initial_R  # average reward
        self.gamma = args.Gamma  # discount factor

    def choose_action(self, s, epsilon):
        if np.random.random() < epsilon:
            return np.random.choice(Action_Num)
        else:
            return tf.argmax(self.active_qnet(s[None, :]), 1)[0].numpy()  # batch size = 1

    def train(self, batch_size=args.Batch_Size):
        # sample from buffer
        s, a, r, s_next = self.buffer.sample(batch_size)

        # calculate target q
        q_next_lazy = self.lazy_qnet(s_next)
        max_q_next_lazy = tf.reduce_max(q_next_lazy, 1, True)
        q_target = r + self.gamma * max_q_next_lazy

        # calculate loss
        with tf.GradientTape() as tape:
            q_active = self.active_qnet(s)
            q_chosen_active = tf.gather(q_active, a, batch_dims=-1)

            td = q_target - q_chosen_active
            loss = tf.reduce_mean(tf.square(td))

        # update R
        grads = tape.gradient(loss, self.active_qnet.trainable_variables)  # gradients
        grads = [tf.clip_by_norm(grad, 10.0) for grad in grads]

        self.active_qnet.optimizer.apply_gradients(zip(grads, self.active_qnet.trainable_variables))

    def update_lazy_q(self):
        # update target q
        for lazy, active in zip(self.lazy_qnet.trainable_variables, self.active_qnet.trainable_variables):
            lazy.assign(active)

    def save_model(self, dir=log_dir_name + '/models'):
        self.lazy_qnet.save_weights(dir + '/' + self.alg + '_lazy_qnet.h5')
        self.active_qnet.save_weights(dir + '/' + self.alg + '_active_qnet.h5')

def train(points=args.Points):
    agent = dqn_agent(args.Max_Epsilon, args.Batch_Size, args.Memory_Size)
    print("============" + agent.alg + "============")

    st = env.reset() / env.AoI_Max  # St
    step = 0
    summary_step = 0

    while summary_step < points:
        env.simulation_user_request()
        a = agent.choose_action(st, agent.epsilon)
        stp1, r = env.step(a)  # St+1, reward
        stp1 = stp1 / env.AoI_Max  # normalize state

        agent.buffer.store((st, a, r[0], stp1))

        st = stp1

        # train
        if step > args.Start_Size:
            agent.train(args.Batch_Size)

        # update target q
        if step % args.Update_Lazy_Step == 0:
            agent.update_lazy_q()

        # evaluate
        if step > args.Start_Size and step % args.Evaluate_Interval == 0:
            feedbacks = evaluate(agent, eps=args.Test_Epsilon)
            greedy_test_mean_r = feedbacks[0]
            aoi_cost = feedbacks[1]
            energy_cost = feedbacks[2]
            agent.save_model()

            # log
            with fw.as_default():
                tf.summary.scalar('greedy005_test_mean_r', greedy_test_mean_r, step=summary_step)
                tf.summary.scalar('R', agent.R, step=summary_step)
                tf.summary.scalar('epsilon', agent.epsilon, step=summary_step)
                tf.summary.scalar('aoi_cost', aoi_cost, step=summary_step)
                tf.summary.scalar('energy_cost', energy_cost, step=summary_step)
                summary_step += 1

        # epsilon decay
        agent.epsilon = max(Epsilon_Decay_Rate * step + args.Max_Epsilon, args.Min_Epsilon)
        step += 1

def evaluate(agent, n_step=args.Test_Step, eps=0.0):
    test_env = AoI_Energy(user_num=args.User, beta2=args.Beta2, seed=args.Seed, request_p=args.Request_P)
    st = test_env.reset() / test_env.AoI_Max
    total_r = 0.0
    aoi_cost = 0.0
    energy_cost = 0.0
    for _ in range(n_step):
        test_env.simulation_user_request()
        a = agent.choose_action(st, eps)
        stp1, r = test_env.step(a)
        st = stp1 / test_env.AoI_Max

        total_r += r[0]
        aoi_cost += r[1]
        energy_cost += r[2]
    return (total_r / n_step, aoi_cost / n_step, energy_cost / n_step)

if __name__ == "__main__":
    train(args.Points)
