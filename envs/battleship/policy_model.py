from envs.battleship.types import State_Battleship, Action_Battleship, Coord
import random


class PolicyModel_Battleship:
    def sample(self, state: State_Battleship) -> Action_Battleship:
        return Action_Battleship(
            coord=Coord(random.randint(0, 9), random.randint(0, 9))
        )

    def rollout(self, state, *args) -> Action_Battleship:
        return self.sample(state)

    def get_all_actions(self, state=None, history=None):
        all_actions = [
            Action_Battleship(coord=Coord(x, y)) for x in range(10) for y in range(10)
        ]
        all_prev_actions = [a for a, o in history]
        for a in all_actions:
            if a in all_prev_actions:
                all_actions.remove(a)
        return all_actions
