from envs.battleship.types import (
    Observation_Battleship,
    Action_Battleship,
    State_Battleship,
)


class ObservationModel_Battleship:
    def sample(
        self, next_state: State_Battleship, action: Action_Battleship
    ) -> Observation_Battleship:
        if next_state.is_occupied(action.coord):
            return Observation_Battleship("hit")
        else:
            return Observation_Battleship("miss")
