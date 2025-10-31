from envs.battleship.types import (
    State_Battleship,
    Action_Battleship,
    Coord,
    Observation_Battleship,
    get_occupation_coords,
)
import random


class PolicyModel_Battleship:
    def rollout(
        self,
        state: State_Battleship,
        history: list[tuple[Action_Battleship, Observation_Battleship]],
    ) -> Action_Battleship:
        return random.choice(self.get_all_actions(state, history))

    def get_all_actions(
        self,
        state: State_Battleship,
        history: list[tuple[Action_Battleship, Observation_Battleship]],
    ) -> list[Action_Battleship]:
        all_actions = self.get_preferred_actions(state)
        output = []
        all_prev_actions = [a for a, o in history]
        for a in all_actions:
            if a not in all_prev_actions:
                output.append(a)
        return output

    def get_preferred_actions(
        self,
        state: State_Battleship,
    ) -> list[Action_Battleship]:
        actions = []
        for ship in state.ships:
            actions.extend(
                [
                    Action_Battleship(coord=coord)
                    for coord in get_occupation_coords(
                        ship.pos, ship.direction, ship.length
                    )
                ]
            )
        return actions
