from envs.tiger.types import Observation_Tiger
import random


class ObservationModel_Tiger:
    def __init__(self, n: int, noise=0.1):
        self.noise = noise
        self.n = n

    def probability(self, observation, next_state, action) -> float:
        if action.name == "listen":
            # heard the correct growl
            if observation.name == next_state.name:
                return 1.0 - self.noise
            else:
                return self.noise / (self.n - 1)
        else:
            return 1 / self.n

    def sample(self, next_state, action) -> Observation_Tiger:
        if action.name == "listen":
            thresh = 1.0 - self.noise
        else:
            thresh = 1 / self.n

        if random.uniform(0, 1) < thresh:
            return Observation_Tiger(next_state.name, self.n)
        else:
            return random.choice(
                [o for o in self.get_all_observations() if o.name != next_state.name]
            )

    def get_all_observations(self):
        """Only need to implement this if you're using
        a solver that needs to enumerate over the observation
        space (e.g. value iteration)"""
        return [Observation_Tiger(f"tiger-{s}", self.n) for s in range(self.n)]
