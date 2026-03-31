from typing import Optional


RL_STYLE_K_TRAINERS = {"zo_rl", "zo_rl_sgd", "zo_rl_adamm", "hizoo_rl"}


def resolve_k_value(trainer_name: Optional[str], k_value: Optional[int]) -> int:
    if k_value is not None:
        if k_value < 1:
            raise ValueError(f"k_value must be >= 1, got {k_value}")
        return k_value

    if trainer_name in RL_STYLE_K_TRAINERS:
        return 10

    return 1
