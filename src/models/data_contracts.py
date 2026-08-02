from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class MatchRecord:
    """Typed representation of a single fixture."""

    date: str
    season: str
    home_team: str
    away_team: str
    home_corners: float
    away_corners: float


@dataclass(frozen=True)
class TeamRating:
    """Typed representation of a team rating row."""

    team: str
    home_attack_rating: float
    away_attack_rating: float
    home_defence_rating: float
    away_defence_rating: float
    home_corner_advantage: float
    away_corner_penalty: float
    overall_attack: float
    overall_defence: float
    tempo_index: float
    corner_difference: float
    corner_balance: float
    opponent_strength_adjustment: float
    sample_size: float
    rating_std: float
    confidence: float
    standard_deviation: float
    consistency_index: float


@dataclass(frozen=True)
class FeatureRecord:
    """Typed representation of a single match feature row."""

    home_last5_corner_for: float
    home_last5_corner_against: float
    away_last5_corner_for: float
    away_last5_corner_against: float
    home_last10_corner_for: float
    away_last10_corner_for: float
    home_attack_rating: float
    away_attack_rating: float
    home_defence_rating: float
    away_defence_rating: float
    home_tempo: float
    away_tempo: float
    home_consistency: float
    away_consistency: float
    expected_total_corner: float
    expected_home_corner: float
    expected_away_corner: float
    rating_difference: float
    tempo_difference: float
    home_advantage: float
    home_std: float
    away_std: float
    combined_std: float
    over85: int
    over95: int
    over105: int
    over115: int
    under85: int
    under95: int
    under105: int
    under115: int


@dataclass(frozen=True)
class PredictionRecord:
    """Typed representation of a single prediction row."""

    expected_home_corners: float
    expected_away_corners: float
    expected_total_corners: float
    over_8: float
    under_8: float
    over_9: float
    under_9: float
    over_10: float
    under_10: float
    over_11: float
    under_11: float
    actual_total_corners: float
