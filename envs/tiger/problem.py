from envs.tiger.types import *
from agent import Agent, Environment
from envs.tiger.observation_model import ObservationModel_Tiger
from envs.tiger.transition_model import TransitionModel_Tiger
from envs.tiger.reward_model import RewardModel_Tiger
from envs.tiger.policy_model import PolicyModel_Tiger


class Problem_Tiger:
    def __init__(
        self, n: int, obs_noise: float, init_true_state: State_Tiger, init_belief
    ) -> None:
        self.agent = Agent(
            init_belief,
            policy_model=PolicyModel_Tiger(n),
            transition_model=TransitionModel_Tiger(n),
            reward_model=RewardModel_Tiger(),
            observation_model=ObservationModel_Tiger(n, obs_noise),
        )
        self.env = Environment(
            init_true_state,
            transition_model=TransitionModel_Tiger(n),
            reward_model=RewardModel_Tiger(),
        )
