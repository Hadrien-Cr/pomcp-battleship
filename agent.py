class Environment:

    def __init__(
        self, init_state, transition_model=None, reward_model=None, blackbox_model=None
    ) -> None:
        self.init_state = init_state
        self.cur_state = init_state
        self.transition_model = transition_model
        self.reward_model = reward_model
        self.blackbox_model = blackbox_model

    def state_transition(self, action, execute=True, discount_factor=1.0):
        next_state, reward, _ = sample_explict_models(
            self.transition_model,
            None,
            self.reward_model,
            self.state,
            action,
            discount_factor=discount_factor,
        )
        if execute:
            self.apply_transition(next_state)
            return reward
        else:
            return next_state, reward

    # def apply_transition(self, next_state):
    #     self.cur_state = next_state

    # def execute(self, action, observation_model):
    #     reward = self.state_transition(action, execute=True)
    #     observation = self.provide_observation(observation_model, action)
    #     return (observation, reward)

    # def provide_observation(self, observation_model, action):
    #     return observation_model.sample(self.state, action)


class Agent:
    def __init__(
        self,
        init_belief,
        policy_model=None,
        transition_model=None,
        observation_model=None,
        reward_model=None,
        blackbox_model=None,
        name=None,
    ) -> None:
        self.init_belief = init_belief
        self.policy_model = policy_model

        self.transition_model = transition_model
        self.observation_model = observation_model
        self.reward_model = reward_model
        self.blackbox_model = blackbox_model
        self.name = name

        # For online planning
        self.cur_belief = init_belief
        self.history = []

    def update_history(self, real_action, real_observation):
        """update_history(self, real_action, real_observation)"""
        self.history += [(real_action, real_observation)]

    def set_belief(self, belief, prior=False) -> None:
        """set_belief(self, belief, prior=False)"""
        print("set belief", belief)
        self.cur_belief = belief
        if prior:
            self.init_belief = belief

    def sample_belief(self):
        """sample_belief(self)
        Returns a state (:class:`State`) sampled from the belief."""
        return self.cur_belief.random()

    def valid_actions(self, state=None, history=None):
        return self.policy_model.get_all_actions(state=state, history=history)
