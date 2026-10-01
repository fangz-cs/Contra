"""Provider-independent components of the CONTRA clarification method."""

from .behavioral import BehavioralResult, CaseEvidence, ExecutionResult, verify_behavior
from .discovery import Candidate, DiscoveryResult, discover_questions

__all__ = [
    "BehavioralResult",
    "Candidate",
    "CaseEvidence",
    "DiscoveryResult",
    "ExecutionResult",
    "discover_questions",
    "verify_behavior",
]
