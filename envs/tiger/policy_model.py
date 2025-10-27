from envs.tiger.types import State_Tiger, Action_Tiger
import random


class PolicyModel_Tiger:
    def __init__(self, n: int):
        self.n = n

    def sample(self, state):
        return random.sample(self.get_all_actions(), 1)[0]

    def rollout(self, state, *args) -> State_Tiger:
        return self.sample(state)

    def get_all_actions(self, state=None, history=None) -> list[Action_Tiger]:
        return [
            Action_Tiger(s, self.n)
            for s in [f"open-{s}" for s in range(self.n)] + ["listen"]
        ]
