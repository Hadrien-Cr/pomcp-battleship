from envs.battleship.types import (
    State_Battleship,
    Action_Battleship,
    Observation_Battleship,
)

from envs.battleship.reward_model import RewardModel_Battleship
from envs.battleship.observation_model import ObservationModel_Battleship
from envs.battleship.transition_model import TransitionModel_Battleship
from envs.battleship.policy_model import PolicyModel_Battleship

from agent import Agent, Environment
import random
import copy


class Problem_Battleship:
    def __init__(self, init_true_state: State_Battleship, init_belief) -> None:
        self.agent = Agent(
            init_belief,
            policy_model=PolicyModel_Battleship(),
            transition_model=TransitionModel_Battleship(),
            reward_model=RewardModel_Battleship(),
            observation_model=ObservationModel_Battleship(),
        )
        self.env = Environment(
            init_true_state,
            transition_model=TransitionModel_Battleship(),
            reward_model=RewardModel_Battleship(),
        )

    def state_transform_func(
        self,
        state: State_Battleship,
        history: list[tuple[Action_Battleship, Observation_Battleship]],
    ) -> State_Battleship:
        r = random.randint(0, 2)

        if r == 0:
            next_state = state._ship_swap()
            while not next_state._is_coherent_with_history(history):
                next_state = state._ship_swap()
            return next_state
        elif r == 1:
            next_state = state._ship_merge()
            while not next_state._is_coherent_with_history(history):
                next_state = state._ship_merge()
            return next_state
        elif r == 2:
            next_state = state._ship_move()
            while not next_state._is_coherent_with_history(history):
                next_state = state._ship_move()
            return next_state
        else:
            return state
