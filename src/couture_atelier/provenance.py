"""Helpers for keeping observed and simulated data distinguishable."""

from enum import StrEnum


class Provenance(StrEnum):
    OBSERVED_PUBLIC = "observed_public"
    DERIVED = "derived"
    SYNTHETIC = "synthetic"
    SCENARIO = "scenario"
