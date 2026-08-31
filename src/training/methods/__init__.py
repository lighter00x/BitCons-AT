from .pgd_at import pgd_at_train
from .trades import trades_train
from .mart import mart_train
from .cons_at import cons_at_train
from .rpat import rpat_train
from .bitmax_at import bitmax_at_train


def get_train_fn(config):
    """
    Get training function for the specified method.

    Args:
        config: Configuration object with 'method' attribute

    Returns:
        Training function for the specified method

    Raises:
        ValueError: If the specified method is not supported
    """
    if config.method in ("pgd_at", "bitplane_at", "quantize_at"):
        return pgd_at_train
    elif config.method == "trades":
        return trades_train
    elif config.method == "mart":
        return mart_train
    elif config.method == "cons_at":
        return cons_at_train
    elif config.method == "rpat":
        return rpat_train
    elif config.method == "bitmax_at":
        return bitmax_at_train
    else:
        available_methods = [
            "pgd_at", "trades", "mart", "cons_at", "rpat", "bitmax_at",
            "bitplane_at", "quantize_at",
        ]
        raise ValueError(
            f"Unknown training method: '{config.method}'. "
            f"Available methods: {', '.join(available_methods)}"
        )
