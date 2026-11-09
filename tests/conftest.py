import numpy as np
import pytest
from mmm.analysis import Params


@pytest.fixture
def params():
    """Synthetic posterior with known structure (3 channels, 400 draws)."""
    rng = np.random.default_rng(0)
    S, C = 400, 3
    return Params(
        channels=["a", "b", "c"],
        alpha=rng.beta(2, 5, (S, C)),
        lam=rng.gamma(4, 0.6, (S, C)),
        beta=np.abs(rng.normal([0.30, 0.10, 0.02], 0.02, (S, C))),
        channel_scale=np.array([100.0, 50.0, 20.0]),
        target_scale=1_000_000.0,
        l_max=8,
        spend_mean=np.array([40.0, 20.0, 10.0]),
        spend_max=np.array([100.0, 50.0, 20.0]),
        n_weeks=104,
    )
