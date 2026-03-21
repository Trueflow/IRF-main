import numpy as np
from modules.Actor.LRN_KNN import LRU_KNN

class Episodic_memory_buffer:
    def __init__(self, args):
        self.ec_buffer = LRU_KNN(args.emdqn_buffer_size, args.emdqn_latent_dim, 'game')
        self.rng = np.random.RandomState(123456)  # deterministic, erase 123456 for stochastic
        self.random_projection = self.rng.normal(loc=0, scale=1. / np.sqrt(args.emdqn_latent_dim),
                                       size=(args.emdqn_latent_dim, args.state_size))
        self.q_episodic_memeory_cwatch = []
        self.args=args
        self.update_counter =0
        self.qecwatch=[]
        self.qec_found=0
        self.update_counter=0

    def update_kdtree(self):
        self.ec_buffer.update_kdtree()

    def peek(self, key, value_decay, modify):
        return self.ec_buffer.peek(key, value_decay, modify)

    def update_ec(self, state, action, reward):
        ep_state = state[0, :] # batch 차원 없애는 것으로 추정
        ep_action = action[0, :]
        ep_reward = reward[0, :]
        Rtd = 0.
        for t in range(episode_batch.max_seq_length - 1, -1, -1):
            s = ep_state[t]
            a = ep_action[t]
            r = ep_reward[t]
            z = np.dot(self.random_projection, s.flatten().cpu())
            Rtd = r + self.args.gamma * Rtd
            z = z.reshape((self.args.emdqn_latent_dim))
            qd = self.ec_buffer.peek(z, Rtd, True)
            if qd == None:  # new action
                self.ec_buffer.add(z, Rtd)
    def hit_probability(self):
        return (1.0 * self.qec_found / self.args.batch_size / self.update_counter)

