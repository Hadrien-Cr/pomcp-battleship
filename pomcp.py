from particles import Particles, sample_generative_model, particle_reinvigoration
import copy
import time
import random
import math
from tqdm import tqdm
from typing import Any
from agent import Agent


class TreeNode:
    def __init__(self):
        self.children = {}

    def __getitem__(self, key):
        return self.children.get(key, None)

    def __setitem__(self, key, value):
        self.children[key] = value

    def __contains__(self, key):
        return key in self.children


class ANDNode(TreeNode):
    """Qnode is a node that has edges labeled by observations
    children is a dictionary observation -> ORNode"""

    def __init__(self, num_visits, value):
        self.num_visits = num_visits
        self.value = value
        self.children = {}

    def __str__(self):
        return "ANDNode" + "(%.3f, %.3f | %s)" % (
            self.num_visits,
            self.value,
            str(self.children.keys()),
        )

    def __repr__(self):
        return self.__str__()


class ORNode(TreeNode):
    """ORNode is a node that has edges labeled by actions
    children is a dictionary action -> ANDNode"""

    def __init__(self, num_visits, **kwargs):
        self.num_visits = num_visits
        self.children = {}

    def __str__(self):
        return "ORNode" + "(%.3f, %.3f | %s)" % (
            self.num_visits,
            self.value,
            str(self.children.keys()),
        )

    def __repr__(self):
        return self.__str__()

    def select_best_action(self):
        """Returns the action of the child with highest value"""
        best_value = float("-inf")
        best_action = None
        for action in self.children:
            if self[action].value > best_value:
                best_action = action
                best_value = self[action].value
        return best_action

    @property
    def value(self):
        best_action = max(self.children, key=lambda action: self.children[action].value)
        return self.children[best_action].value


class RootORNode(ORNode):
    """Root of the search tree"""

    def __init__(self, num_visits, history):
        ORNode.__init__(self, num_visits)
        self.history = history


class ORNodeParticles(ORNode):
    """POMCP's ORNode maintains particle belief"""

    def __init__(self, num_visits, belief=Particles([])):
        self.num_visits = num_visits
        self.belief = belief
        self.children = {}  # a -> QNode

    def __str__(self):
        return "ORNode(%.3f, %.3f, %d | %s)" % (
            self.num_visits,
            self.value,
            len(self.belief),
            str(self.children.keys()),
        )

    def __repr__(self):
        return self.__str__()


class RootORNodeParticles(RootORNode):
    def __init__(self, num_visits, history, belief=Particles([])):
        RootORNode.__init__(self, num_visits, history)
        self.belief = belief


class RolloutPolicy:
    def rollout(self, state, history):
        """rollout(self, State state, tuple history=None)"""
        pass


###################### POUCT #########################


class POUCT:
    """POUCT (Partially Observable UCT) :cite:`silver2010monte` is presented in the POMCP
    paper as an extension of the UCT algorithm to partially-observable domains
    that combines MCTS and UCB1 for action selection.

    POUCT only works for problems with action space that can be enumerated.

    Args:
        max_depth (int): Depth of the MCTS tree. Default: 5.
        planning_time (float), amount of time given to each planning step (seconds). Default: -1.
            if negative, then planning terminates when number of simulations `num_sims` reached.
            If both `num_sims` and `planning_time` are negative, then the planner will run for 1 second.
        num_sims (int): Number of simulations for each planning step. If negative,
            then will terminate when planning_time is reached.
            If both `num_sims` and `planning_time` are negative, then the planner will run for 1 second.
        rollout_policy (RolloutPolicy): rollout policy. Default: RandomRollout.
        action_prior (ActionPrior): a prior over preferred actions given state and history.

    """

    def __init__(
        self,
        agent: Agent,
        max_depth=5,
        planning_time=-1.0,
        num_sims=-1,
        discount_factor=0.9,
        c_UCB=math.sqrt(2),
        num_visits_init=0,
        value_init=0,
        rollout_policy=None,
    ):
        self._max_depth = max_depth
        self._planning_time = planning_time
        self._num_sims = num_sims

        self.num_visits_init = num_visits_init
        self.value_init = value_init
        self.rollout_policy = rollout_policy

        self.discount_factor = discount_factor
        self.c_UCB = c_UCB

        # to simplify function calls; plan only for one agent at a time
        self.agent = agent
        self._last_num_sims = -1
        self._last_planning_time = -1

    def plan(self) -> Any:
        if not hasattr(self.agent, "tree"):
            setattr(self.agent, "tree", None)
        action, time_taken, sims_count = self._search()
        self._last_num_sims = sims_count
        self._last_planning_time = time_taken
        return action

    def _do_simulate(self, state):
        self._simulate(state, self.agent.history, self.agent.tree, None, None, 0)

    def update(self, agent, real_action, real_observation) -> None:
        if (
            real_action not in agent.tree
            or real_observation not in agent.tree[real_action]
        ):
            agent.tree = (
                None  # replan, if real action or observation differs from all branches
            )
        elif agent.tree[real_action][real_observation] is not None:
            # Update the tree (prune)
            ornode = agent.tree[real_action][real_observation]
            children = ornode.children
            agent.tree = RootORNode(ornode.num_visits, agent.history)
            agent.tree.children = children
        else:
            raise ValueError("Unexpected state; child should not be None")

    def _expand_ornode(self, ornode, history, state) -> None:
        for action in self.agent.valid_actions(state=state, history=history):
            if ornode[action] is None:
                history_action_node = ANDNode(self.num_visits_init, self.value_init)
                ornode[action] = history_action_node

    def _search(self) -> tuple:
        sims_count = 0
        start_time = time.time()

        while not self._should_stop(sims_count, start_time):
            state = self.agent.sample_belief()
            self._do_simulate(state)
            sims_count += 1

        best_action = self.agent.tree.select_best_action()  # type: ignore
        time_taken = time.time() - start_time
        return best_action, time_taken, sims_count

    def _should_stop(self, sims_count, start_time):
        time_taken = time.time() - start_time
        if self._num_sims > 0:
            return sims_count >= self._num_sims
        else:
            return time_taken > self._planning_time

    def _simulate(self, state, history, root, parent, observation, depth) -> float:

        # if g^d < eps
        if depth > self._max_depth:
            return 0

        # if h not in T
        if root is None:
            if self.agent.tree is None:  # type: ignore
                root = self._get_ORNode(root=True)
                self.agent.tree = root  # type: ignore
                if self.agent.tree.history != self.agent.history:  # type: ignore
                    raise ValueError("Unable to plan for the given history.")
            else:
                root = self._get_ORNode()
            if parent is not None:
                parent[observation] = root
            self._expand_ornode(root, history, state)
            rollout_reward = self._rollout(state, history, depth)
            return rollout_reward

        action = self._ucb(root)
        next_state, observation, reward = sample_generative_model(
            self.agent, state, action
        )

        total_reward = reward + self.discount_factor * self._simulate(
            state=next_state,
            history=history + [(action, observation)],
            root=root[action][observation],
            parent=root[action],
            observation=observation,
            depth=depth + 1,
        )
        root.num_visits += 1
        root[action].num_visits += 1
        root[action].value = root[action].value + (
            total_reward - root[action].value
        ) / (root[action].num_visits)
        return total_reward

    def _rollout(self, state, history, depth) -> float:

        discount = 1.0
        total_discounted_reward = 0

        while depth < self._max_depth:
            action = self.rollout_policy.rollout(state, history)  # type: ignore
            next_state, observation, reward = sample_generative_model(
                self.agent, state, action
            )
            history = history + [(action, observation)]
            depth += 1
            total_discounted_reward += reward * discount
            discount *= self.discount_factor
            state = next_state
        return total_discounted_reward

    def _ucb(self, root) -> Any:
        """UCB1"""
        best_action, best_value = None, float("-inf")
        for action in root.children:
            if root[action].num_visits == 0:
                val = float("inf")
            else:
                val = root[action].value + self.c_UCB * math.sqrt(
                    math.log(root.num_visits + 1) / root[action].num_visits
                )
            if val > best_value:
                best_action = action
                best_value = val
        return best_action

    def _get_ORNode(self, root=False, **kwargs) -> ORNode:
        """Returns a ORNode with default values; The function naming makes it clear
        that this function is about creating a ORNode object."""
        if root:
            return RootORNode(self.num_visits_init, self.agent.history)

        else:
            return ORNode(self.num_visits_init)


###################### POMCP #########################


class POMCP(POUCT):
    """POMCP is POUCT + particle belief representation.
    This POMCP version only works for problems
    with action space that can be enumerated."""

    def __init__(
        self,
        agent: Agent,
        max_depth=5,
        planning_time=-1.0,
        num_sims=-1,
        discount_factor=0.9,
        c_UCB=math.sqrt(2),
        num_visits_init=0,
        value_init=0,
        rollout_policy=None,
        action_prior=None,
        show_progress=False,
        pbar_update_interval=5,
    ) -> None:
        super().__init__(
            agent=agent,
            max_depth=max_depth,
            planning_time=planning_time,
            num_sims=num_sims,
            discount_factor=discount_factor,
            c_UCB=c_UCB,
            num_visits_init=num_visits_init,
            value_init=value_init,
            rollout_policy=rollout_policy,
        )

    def update(
        self, agent, real_action, real_observation, state_transform_func
    ) -> None:

        if not isinstance(agent.cur_belief, Particles):
            raise TypeError(
                "agent's belief is not represented in particles.\n"
                "POMCP not usable. Please convert it to particles."
            )
        if not hasattr(agent, "tree"):
            raise ValueError("Warning: agent does not have tree. Have you planned yet?")

        if agent.tree[real_action][real_observation] is None:
            # Never anticipated the real_observation. No reinvigoration can happen.
            raise ValueError("Particle deprivation.")
        # Update the tree; Reinvigorate the tree's belief and use it
        # as the updated belief for the agent.

        ornode = agent.tree[real_action][real_observation]
        assert ornode.num_visits > 0  # should have been visited
        children = ornode.children
        agent.tree = RootORNodeParticles(
            agent.tree[real_action][real_observation].num_visits,
            agent.history,
            ornode.belief,
        )
        agent.tree.children = children

        tree_belief = agent.tree.belief

        agent.set_belief(
            particle_reinvigoration(
                tree_belief,
                len(agent.init_belief.particles),
                history=agent.history,
                state_transform_func=state_transform_func,
            )
        )
        # If observation was never encountered in simulation, then tree will be None;
        # particle reinvigoration will occur.
        if agent.tree is not None:
            agent.tree.belief = copy.deepcopy(agent.cur_belief)

    def _simulate(self, state, history, root, parent, observation, depth) -> float:
        total_reward = POUCT._simulate(
            self, state, history, root, parent, observation, depth
        )
        if depth == 1 and root is not None:
            root.belief.add(state)  # belief update happens as simulation goes.
        return total_reward

    def _get_ORNode(self, root=False, **kwargs) -> ORNode:
        """Returns a ORNode with default values; The function naming makes it clear
        that this function is about creating a ORNode object."""
        if root:
            # agent cannot be None.
            return RootORNodeParticles(
                self.num_visits_init,
                self.agent.history,
                belief=copy.deepcopy(self.agent.cur_belief),  # type: ignore
            )
        else:
            return ORNodeParticles(self.num_visits_init, belief=Particles([]))
