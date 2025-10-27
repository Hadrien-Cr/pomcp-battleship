import numpy as np
import random


class Histogram:
    """
    Histogram representation of a probability distribution.

    __init__(self, histogram)

    Args:
        histogram (dict) is a dictionary mapping from
            variable value to probability
    """

    def __init__(self, histogram):
        """`histogram` is a dictionary mapping from
        variable value to probability"""
        if not (isinstance(histogram, dict)):
            raise ValueError(
                "Unsupported histogram representation! %s" % str(type(histogram))
            )
        self._histogram = histogram

    @property
    def histogram(self):
        """histogram(self)"""
        return self._histogram

    def __str__(self):
        return str(self._histogram)

    def __len__(self):
        return len(self._histogram)

    def __getitem__(self, value):
        """__getitem__(self, value)
        Returns the probability of `value`."""
        if value in self._histogram:
            return self._histogram[value]
        else:
            return 0

    def __setitem__(self, value, prob):
        """__setitem__(self, value, prob)
        Sets probability of value to `prob`."""
        self._histogram[value] = prob

    def __eq__(self, other):
        if not isinstance(other, Histogram):
            return False
        else:
            return self.histogram == other.histogram

    def __iter__(self):
        return iter(self._histogram)

    def mpe(self):
        """mpe(self)
        Returns the most likely value of the variable.
        """
        return max(self._histogram, key=self._histogram.get)

    def random(self):
        """
        random(self)
        Randomly sample a value based on the probability
        in the histogram"""
        candidates = list(self._histogram.keys())
        prob_dist = []
        for value in candidates:
            prob_dist.append(self._histogram[value])
        return np.random.choice(candidates, 1, p=prob_dist)[0]

    def get_histogram(self):
        """get_histogram(self)
        Returns a dictionary from value to probability of the histogram"""
        return self._histogram

    # Deprecated; it's assuming non-log probabilities
    def is_normalized(self, epsilon=1e-9):
        """Returns true if this distribution is normalized"""
        prob_sum = sum(self._histogram[state] for state in self._histogram)
        return abs(1.0 - prob_sum) < epsilon


def update_histogram_belief(
    current_histogram,
    real_action,
    real_observation,
    observation_model,
    transition_model,
    oargs={},
    targs={},
    normalize=True,
    static_transition=False,
):
    new_histogram = {}  # state space still the same.
    total_prob = 0

    for next_state in next_state_space:
        observation_prob = observation_model.probability(
            real_observation, next_state, real_action, **oargs
        )
        if not static_transition:
            transition_prob = 0
            for state in current_histogram:
                transition_prob += (
                    transition_model.probability(
                        next_state, state, real_action, **targs
                    )
                    * current_histogram[state]
                )
        else:
            transition_prob = current_histogram[next_state]

        new_histogram[next_state] = observation_prob * transition_prob
        total_prob += new_histogram[next_state]

    # Normalize
    if normalize:
        for state in new_histogram:
            if total_prob > 0:
                new_histogram[state] /= total_prob
    return Histogram(new_histogram)
