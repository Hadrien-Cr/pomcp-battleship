from envs.battleship.types import (
    State_Battleship,
    Action_Battleship,
    Coord,
    Observation_Battleship,
    get_occupation_coords,
)
import random


class PolicyModel_Battleship:
    def sample(self, state: State_Battleship) -> Action_Battleship:
        return random.choice(self.get_all_actions(state, []))

    def rollout(self, state, *args) -> Action_Battleship:
        return self.sample(state)

    def get_all_actions(
        self,
        state: State_Battleship,
        history: list[tuple[Action_Battleship, Observation_Battleship]],
    ) -> list[Action_Battleship]:
        all_actions = [
            Action_Battleship(coord=Coord(x, y)) for x in range(10) for y in range(10)
        ]
        all_prev_actions = [a for a, o in history]
        for a in self.get_preferred_actions(state):
            if a in all_prev_actions:
                all_actions.remove(a)
        return all_actions

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
