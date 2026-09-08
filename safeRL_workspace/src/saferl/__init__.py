"""New research-only safeRL package.

The historical notebook and ``scripts.common`` compatibility layer are not
imported here. New algorithms should depend on explicit configuration and
small, contract-tested modules in this package.
"""

from .config import (
    ActionConfig,
    EvaluationProtocolConfig,
    FrequencyConfig,
    LanelessResearchConfig,
    MINIMUM_ENCLOSING_ELLIPSE_A_M,
    MINIMUM_ENCLOSING_ELLIPSE_B_M,
    ObservationConfig,
    REFERENCE_VEHICLE_LENGTH_M,
    REFERENCE_VEHICLE_WIDTH_M,
    RewardConfig,
    SafetyConfig,
    DEFAULT_LANELESS_RESEARCH_CONFIG,
)

__all__ = [
    "ActionConfig",
    "EvaluationProtocolConfig",
    "FrequencyConfig",
    "LanelessResearchConfig",
    "MINIMUM_ENCLOSING_ELLIPSE_A_M",
    "MINIMUM_ENCLOSING_ELLIPSE_B_M",
    "ObservationConfig",
    "REFERENCE_VEHICLE_LENGTH_M",
    "REFERENCE_VEHICLE_WIDTH_M",
    "RewardConfig",
    "SafetyConfig",
    "DEFAULT_LANELESS_RESEARCH_CONFIG",
]
