"""P6 deep RL. Lazy import keeps PyTorch optional for the other agents."""


def __getattr__(name: str):
    if name == "DQNAgent":
        from .agent import DQNAgent
        return DQNAgent
    raise AttributeError(name)
