from abc import ABC, abstractmethod


class EpsilonScheduler(ABC):
    """Abstract base class for epsilon scheduling strategies."""

    @abstractmethod
    def get_epsilon(self, episode: int) -> float:
        """Get epsilon value for the given episode.

        Args:
            episode: Current episode number

        Returns:
            Epsilon value between 0 and 1
        """
        pass

    @abstractmethod
    def __repr__(self) -> str:
        """String representation of the scheduler."""
        pass


class LinearEpsilonScheduler(EpsilonScheduler):
    """Linear decay epsilon scheduler."""

    def __init__(
        self,
        start_epsilon: float = 1.0,
        end_epsilon: float = 0.05,
        decay_episodes: int = 1000,
    ):
        self.start_epsilon = start_epsilon
        self.end_epsilon = end_epsilon
        self.decay_episodes = decay_episodes

    def get_epsilon(self, episode: int) -> float:
        """Linear decay from start_epsilon to end_epsilon over decay_episodes."""
        return max(
            self.end_epsilon,
            self.start_epsilon
            - (self.start_epsilon - self.end_epsilon) * (episode / self.decay_episodes),
        )

    def __repr__(self) -> str:
        return f"LinearEpsilonScheduler(start_epsilon={self.start_epsilon}, end_epsilon={self.end_epsilon}, decay_episodes={self.decay_episodes})"

    def __str__(self) -> str:
        return f"Linear[{self.start_epsilon}, {self.end_epsilon}, {self.decay_episodes}]"

class ExponentialEpsilonScheduler(EpsilonScheduler):
    """Exponential decay epsilon scheduler."""

    def __init__(
        self,
        start_epsilon: float = 1.0,
        end_epsilon: float = 0.05,
        decay_rate: float = 0.995,
    ):
        self.start_epsilon = start_epsilon
        self.end_epsilon = end_epsilon
        self.decay_rate = decay_rate

    def get_epsilon(self, episode: int) -> float:
        """Exponential decay: epsilon = start_epsilon * (decay_rate ^ episode)."""
        return max(self.end_epsilon, self.start_epsilon * (self.decay_rate**episode))

    def __repr__(self) -> str:
        return f"ExponentialEpsilonScheduler(start_epsilon={self.start_epsilon}, end_epsilon={self.end_epsilon}, decay_rate={self.decay_rate})"


class ConstantEpsilonScheduler(EpsilonScheduler):
    """Constant epsilon scheduler (no decay)."""

    def __init__(self, epsilon: float = 0.1):
        self.epsilon = epsilon

    def get_epsilon(self, episode: int) -> float:
        """Return constant epsilon value."""
        return self.epsilon

    def __repr__(self) -> str:
        return f"ConstantEpsilonScheduler(epsilon={self.epsilon})"


class StepEpsilonScheduler(EpsilonScheduler):
    """Step-wise epsilon decay scheduler."""

    def __init__(
        self,
        start_epsilon: float = 1.0,
        end_epsilon: float = 0.05,
        step_size: int = 100,
        decay_factor: float = 0.9,
    ):
        self.start_epsilon = start_epsilon
        self.end_epsilon = end_epsilon
        self.step_size = step_size
        self.decay_factor = decay_factor

    def get_epsilon(self, episode: int) -> float:
        """Step-wise decay: epsilon decreases by decay_factor every step_size episodes."""
        steps = episode // self.step_size
        epsilon = self.start_epsilon * (self.decay_factor**steps)
        return max(self.end_epsilon, epsilon)

    def __repr__(self) -> str:
        return f"StepEpsilonScheduler(start_epsilon={self.start_epsilon}, end_epsilon={self.end_epsilon}, step_size={self.step_size}, decay_factor={self.decay_factor})"