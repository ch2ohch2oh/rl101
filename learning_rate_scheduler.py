from abc import ABC, abstractmethod


class LearningRateScheduler(ABC):
    """Abstract base class for learning rate scheduling strategies."""

    @abstractmethod
    def get_learning_rate(self, episode: int) -> float:
        """Get learning rate value for the given episode.

        Args:
            episode: Current episode number

        Returns:
            Learning rate value
        """
        pass

    @abstractmethod
    def __repr__(self) -> str:
        """String representation of the scheduler."""
        pass


class ConstantLearningRateScheduler(LearningRateScheduler):
    """Constant learning rate scheduler (no decay)."""

    def __init__(self, learning_rate: float = 0.001):
        self.learning_rate = learning_rate

    def get_learning_rate(self, episode: int) -> float:
        """Return constant learning rate value."""
        return self.learning_rate

    def __repr__(self) -> str:
        return f"ConstantLearningRateScheduler(learning_rate={self.learning_rate})"

    def __str__(self) -> str:
        return f"Constant[{self.learning_rate}]"


class LinearLearningRateScheduler(LearningRateScheduler):
    """Linear decay learning rate scheduler."""

    def __init__(
        self,
        start_lr: float = 0.01,
        end_lr: float = 0.0001,
        decay_episodes: int = 1000,
    ):
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.decay_episodes = decay_episodes

    def get_learning_rate(self, episode: int) -> float:
        """Linear decay from start_lr to end_lr over decay_episodes."""
        return max(
            self.end_lr,
            self.start_lr
            - (self.start_lr - self.end_lr) * (episode / self.decay_episodes),
        )

    def __repr__(self) -> str:
        return f"LinearLearningRateScheduler(start_lr={self.start_lr}, end_lr={self.end_lr}, decay_episodes={self.decay_episodes})"

    def __str__(self) -> str:
        return f"Linear[{self.start_lr}, {self.end_lr}, {self.decay_episodes}]"


class ExponentialLearningRateScheduler(LearningRateScheduler):
    """Exponential decay learning rate scheduler."""

    def __init__(
        self,
        start_lr: float = 0.01,
        end_lr: float = 0.0001,
        decay_rate: float = 0.999,
    ):
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.decay_rate = decay_rate

    def get_learning_rate(self, episode: int) -> float:
        """Exponential decay: lr = start_lr * (decay_rate ^ episode)."""
        return max(self.end_lr, self.start_lr * (self.decay_rate**episode))

    def __repr__(self) -> str:
        return f"ExponentialLearningRateScheduler(start_lr={self.start_lr}, end_lr={self.end_lr}, decay_rate={self.decay_rate})"

    def __str__(self) -> str:
        return f"Exponential[{self.start_lr}, {self.end_lr}, {self.decay_rate}]"


class StepLearningRateScheduler(LearningRateScheduler):
    """Step-wise learning rate decay scheduler."""

    def __init__(
        self,
        start_lr: float = 0.01,
        end_lr: float = 0.0001,
        step_size: int = 100,
        decay_factor: float = 0.9,
    ):
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.step_size = step_size
        self.decay_factor = decay_factor

    def get_learning_rate(self, episode: int) -> float:
        """Step-wise decay: lr decreases by decay_factor every step_size episodes."""
        steps = episode // self.step_size
        lr = self.start_lr * (self.decay_factor**steps)
        return max(self.end_lr, lr)

    def __repr__(self) -> str:
        return f"StepLearningRateScheduler(start_lr={self.start_lr}, end_lr={self.end_lr}, step_size={self.step_size}, decay_factor={self.decay_factor})"

    def __str__(self) -> str:
        return f"Step[{self.start_lr}, {self.end_lr}, {self.step_size}, {self.decay_factor}]"
