from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class Config:
    # A VN enters the ranking only with at least this many votes (VNDB's own
    # top list uses the same threshold).
    min_vote: int = 30
    # A pair (A, B) only counts when at least this many users voted on both.
    min_common_vote: int = 5
    # Per VN, how many opponents to keep for each "head-to-head" category.
    neighbors_per_category: int = 10
    # Skip the rankit-based "scientific ranking" methods (they are the slowest).
    skip_rankit: bool = False
    # Users need this many votes on ranked VNs to get a user page and
    # recommendations (all votes still count towards the rankings).
    min_user_votes: int = 5
    # Skip user pages / recommendations / similar VNs (collaborative filtering).
    skip_users: bool = False
    # Seed for anything stochastic, so snapshots are reproducible.
    seed: int = 0

    def as_dict(self) -> dict:
        return asdict(self)
