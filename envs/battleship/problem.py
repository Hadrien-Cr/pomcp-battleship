from envs.battleship.types import State_Battleship
from envs.battleship.reward_model import RewardModel_Battleship
from envs.battleship.observation_model import ObservationModel_Battleship
from envs.battleship.transition_model import TransitionModel_Battleship
from envs.battleship.policy_model import PolicyModel_Battleship

from agent import Agent, Environment


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
