from envs.tiger.types import State_Tiger, Action_Tiger
import random


class TransitionModel_Tiger:
    def __init__(self, n: int):
        self.n = n

    def probability(
        self, next_state: State_Tiger, state: State_Tiger, action: Action_Tiger
    ) -> float:
        if action.name.startswith("open"):
            return 1 / self.n
        else:
            if next_state.name == state.name:
                return 1.0
            else:
                return 0

    def sample(self, state, action) -> State_Tiger:
        if action.name.startswith("open"):
            return random.choice(self.get_all_states())
        else:
            return state

    def get_all_states(self):
        return [State_Tiger(f"tiger-{s}", self.n) for s in range(self.n)]
