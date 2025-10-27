import random
from generator import Histogram
import copy


class Particles:

    def __init__(
        self, particles, approx_method="none", distance_func=None, frozen=False
    ):
        self.particles = particles

        self._hist = self.get_histogram()
        self._hist_valid = True

        self._approx_method = approx_method
        self._distance_func = distance_func

    def __str__(self):
        return f"Particles({self.get_histogram()}),n={len(self.particles)})"

    def __len__(self) -> int:
        return len(self.particles)

    def __eq__(self, other):
        if isinstance(other, Particles):
            return self._hist == other.hist
        return False

    def __getitem__(self, value):
        """Returns the probability of `value`; normalized"""
        if len(self.particles) == 0:
            raise ValueError("Particles is empty.")

        if not self._hist_valid:
            self._hist = self.get_histogram()
            self._hist_valid = True

        if value in self._hist:
            return self._hist[value]
        else:
            if self._approx_method == "none":
                return 0.0
            elif self._approx_method == "nearest":
                nearest_dist = float("inf")
                nearest = self.particles[0]
                for s in self.particles[1:]:
                    dist = self._distance_func(s, nearest)
                    if dist < nearest_dist:
                        nearest_dist = dist
                        nearest = s
                return self[nearest]
            else:
                raise ValueError("Cannot handle approx_method:", self._approx_method)

    def __setitem__(self, value, prob):
        """
        The particle belief does not support assigning an exact probability to a value.
        """
        raise NotImplementedError

    def mpe(self):
        if not self._hist_valid:
            self._hist = self.get_histogram()
            self._hist_valid = True
        return max(self._hist, key=self._hist.get)

    def __iter__(self):
        return iter(self.particles)

    def condense(self):
        """
        Returns a new set of weighted particles with unique values
        and weights aggregated (taken average).
        """
        return Particles.from_histogram(self.get_histogram())

    def add(self, particle):
        """add(self, particle)
        particle: just a value"""
        self.particles.append(particle)
        self._hist_valid = False

    def get_abstraction(self, state_mapper):
        """get_abstraction(self, state_mapper)
        feeds all particles through a state abstraction function.
        Or generally, it could be any function.
        """
        particles = [state_mapper(s) for s in self.particles]
        return particles

    @classmethod
    def from_histogram(cls, histogram, numparticles=1000):
        """Given a pomdp_py.Histogram return a particle representation of it,
        which is an approximation"""
        particles = []

        for _ in range(numparticles):
            particles.append(histogram.random())
        return Particles(particles)

    def get_histogram(self) -> Histogram:
        hist = {}
        for s in self.particles:
            hist[s] = hist.get(s, 0) + 1
        for s in hist:
            hist[s] = hist[s] / len(self.particles)
        return Histogram(hist)

    def random(self):
        """Samples a value based on the particles"""
        if len(self.particles) > 0:
            return random.choice(self.particles)
        else:
            return None


def sample_generative_model(agent, state, action, discount_factor=1.0) -> tuple:
    assert not hasattr(action, "policy")

    result = sample_explict_models(
        agent.transition_model,
        agent.observation_model,
        agent.reward_model,
        state,
        action,
        agent.history,
        discount_factor,
    )
    return result


def sample_explict_models(
    transition_model,
    observation_model,
    reward_model,
    state,
    action,
    history=None,
    discount_factor=1.0,
) -> tuple:

    next_state = transition_model.sample(state, action)
    reward = reward_model.sample(state, action, history, next_state)

    if observation_model is not None:
        observation = observation_model.sample(next_state, action)
        return next_state, observation, reward
    else:
        return next_state, reward


def particle_reinvigoration(
    particles: Particles, numparticles, state_transform_func=None
) -> Particles:

    # If not enough particles, introduce artificial noise to existing particles (reinvigoration)
    newparticles = copy.deepcopy(particles)
    if len(newparticles) == 0:
        raise ValueError("Particle deprivation.")

    if len(newparticles) > numparticles:
        return newparticles

    while len(newparticles) < numparticles:
        # need to make a copy otherwise the transform affects states in 'particles'
        next_state = copy.deepcopy(particles.random())
        # Add artificial noise
        if state_transform_func is not None:
            next_state = state_transform_func(next_state)
        newparticles.add(next_state)

    return newparticles
