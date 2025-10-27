from envs.tiger.types import Action_Tiger, State_Tiger, Observation_Tiger


class RewardModel_Tiger:
    def sample(
        self,
        state: State_Tiger,
        action: Action_Tiger,
        history: list[tuple[Action_Tiger, Observation_Tiger]],
        next_state: State_Tiger,
    ) -> float:
        if action.name == "listen":
            return -1
        else:
            if state.name.split("-")[1] != action.name.split("-")[1]:
                return 10
            else:
                return -100
