class epsilon_schedule:
    def __init__(self, args):
        self.start = args.start
        self.finish = args.finish
        self.time_length = args.num_episode # 200 episode
        self.delta = (self.start - self.finish) / self.time_length
        self.epsilon = self.start

    def init_schedule(self, train):
        if not train:
            self.start = 0.2
            self.epsilon = 0.2
            self.delta = (0.2 - self.finish) / 100

    def update_epsilon(self, episode):
        self.epsilon = max(self.finish, self.start - self.delta * episode)
        return max(self.finish, self.start - self.delta * episode)