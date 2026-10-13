"""Errors that can be recorded without losing the first contract mismatch."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ContractIssue:
    code: str
    path: str
    message: str
    expected: Any = None
    actual: Any = None


class ContractError(ValueError):
    def __init__(self, issues: tuple[ContractIssue, ...]):
        self.issues = issues
        super().__init__("; ".join(f"{item.path}: {item.message}" for item in issues))


class UnsupportedCapability(RuntimeError):
    """A requested runtime capability is unavailable; never substitute another mode."""
