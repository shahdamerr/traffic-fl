"""Model factory — returns the correct model instance based on config.MODEL_TYPE.

All FL scripts and training loops should use build_model() instead of
importing GRUForecaster or TSMixer directly. This allows switching the
entire pipeline by changing a single line in config.py.

Usage:
    from models.factory import build_model, get_model_kwargs
    model_kwargs = get_model_kwargs(horizon=12)
    model = build_model(model_kwargs)
"""
from config import (
    MODEL_TYPE,
    HIDDEN_SIZE, NUM_LAYERS, DROPOUT,
    TSMIXER_N_BLOCKS, TSMIXER_FF_DIM, TSMIXER_DROPOUT,
)


def get_model_kwargs(horizon: int = 12, seq_len: int = 12) -> dict:
    """Return the hyperparameter dict for the currently configured model.

    Args:
        horizon: forecast horizon (number of steps)
        seq_len: input sequence length

    Returns:
        dict of kwargs to pass to build_model()
    """
    if MODEL_TYPE == "tsmixer":
        return {
            "model_type": "tsmixer",
            "seq_len": seq_len,
            "horizon": horizon,
            "n_blocks": TSMIXER_N_BLOCKS,
            "ff_dim": TSMIXER_FF_DIM,
            "dropout": TSMIXER_DROPOUT,
            "n_features": 1,
        }
    elif MODEL_TYPE == "gru":
        return {
            "model_type": "gru",
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS,
            "horizon": horizon,
            "dropout": DROPOUT,
        }
    elif MODEL_TYPE == "gru_seq2seq":
        return {
            "model_type": "gru_seq2seq",
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS,
            "horizon": horizon,
            "dropout": DROPOUT,
        }
    else:
        raise ValueError(f"Unknown MODEL_TYPE: '{MODEL_TYPE}'. Choose 'tsmixer', 'gru', or 'gru_seq2seq'.")


def build_model(model_kwargs: dict):
    """Instantiate and return the model specified in model_kwargs.

    Args:
        model_kwargs: dict returned by get_model_kwargs()

    Returns:
        nn.Module instance (TSMixer or GRUForecaster)
    """
    model_type = model_kwargs.get("model_type", MODEL_TYPE)
    kwargs = {k: v for k, v in model_kwargs.items() if k != "model_type"}

    if model_type == "tsmixer":
        from models.tsmixer import TSMixer
        return TSMixer(**kwargs)
    elif model_type == "gru":
        from models.gru_forecaster import GRUForecaster
        return GRUForecaster(**kwargs)
    elif model_type == "gru_seq2seq":
        from models.gru_seq2seq import GRUSeq2Seq
        return GRUSeq2Seq(**kwargs)
    else:
        raise ValueError(f"Unknown model_type: '{model_type}'")
