from envs.battleship.types import State_Battleship, Action_Battleship


class TransitionModel_Battleship:
    def probability(
        self,
        next_state: State_Battleship,
        state: State_Battleship,
        action: Action_Battleship,
    ) -> float:
        return float(next_state == state)

    def sample(self, state, action) -> State_Battleship:
        return state
