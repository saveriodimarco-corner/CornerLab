from __future__ import annotations

import numpy as np
import pandas as pd

from src.research.premier_league_forward_model import (
    PremierLeagueForwardModel,
)


class FakeModel:
    def predict(self, frame):
        return np.asarray(
            [10.0] * len(frame),
            dtype=float,
        )


def test_forward_model_applies_frozen_bias_correction():
    model = PremierLeagueForwardModel(
        base_model=FakeModel(),
        bias_correction=0.4,
        feature_names=("feature_a",),
    )

    frame = pd.DataFrame(
        {"feature_a": [1.0, 2.0]}
    )

    prediction = model.predict(frame)

    assert prediction.tolist() == [9.6, 9.6]
    assert model.feature_names_in_.tolist() == [
        "feature_a"
    ]

def test_loader_verifies_hash_and_safety_contract(
    tmp_path: Path,
) -> None:
    import hashlib
    import json
    import pickle

    from src.research.premier_league_forward_model import (
        ELIGIBLE_LINES,
        FEATURES,
        PremierLeagueForwardModel,
        load_forward_model,
    )

    cache_dir = (
        tmp_path
        / "data"
        / "provider_cache"
        / "premier_league_validation"
    )
    cache_dir.mkdir(parents=True)

    artifact_path = (
        cache_dir
        / "premier_league_forward_model.pkl"
    )
    model = PremierLeagueForwardModel(
        base_model=None,
        bias_correction=0.25,
    )
    artifact_path.write_bytes(
        pickle.dumps(model)
    )

    artifact_hash = hashlib.sha256(
        artifact_path.read_bytes()
    ).hexdigest()

    manifest = {
        "mode": "forward_validation_only",
        "feature_names": list(FEATURES),
        "eligible_lines": list(ELIGIBLE_LINES),
        "artifact_path": str(artifact_path),
        "artifact_hash": artifact_hash,
        "paper_trading_enabled": True,
        "telegram_simulation_enabled": False,
        "real_betting_enabled": False,
        "bankroll_real_enabled": False,
    }

    manifest_path = (
        cache_dir
        / "premier_league_forward_manifest.json"
    )
    manifest_path.write_text(
        json.dumps(manifest) + "\n",
        encoding="utf-8",
    )

    loaded_model, loaded_manifest = (
        load_forward_model(tmp_path)
    )

    assert isinstance(
        loaded_model,
        PremierLeagueForwardModel,
    )
    assert loaded_model.bias_correction == 0.25
    assert loaded_manifest["artifact_hash"] == (
        artifact_hash
    )

    artifact_path.write_bytes(b"altered")

    try:
        load_forward_model(tmp_path)
    except ValueError as exc:
        assert "Hash artefatto" in str(exc)
    else:
        raise AssertionError(
            "Artefatto alterato accettato"
        )
