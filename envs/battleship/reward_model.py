from envs.battleship.types import (
    Action_Battleship,
    State_Battleship,
    Observation_Battleship,
)


class RewardModel_Battleship:
    def sample(
        self,
        state: State_Battleship,
        action: Action_Battleship,
        history: list[tuple[Action_Battleship, Observation_Battleship]],
        next_state: State_Battleship,
    ) -> float:
        coords_occupied = state.get_all_occupied()

        if len(coords_occupied) > len(history) + 1:
            return -1
        else:
            if set(coords_occupied).issubset(
                [a.coord for a, _ in history] + [action.coord]
            ):
                return 100
            else:
                return -1
