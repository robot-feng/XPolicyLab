"""Registries store factories by name and import implementations only on demand."""

from __future__ import annotations

import importlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Entry:
    name: str
    factory: str
    capabilities: tuple[str, ...] = ()
    origin: str = "xpolicy"

    def load(self):
        module, separator, attribute = self.factory.partition(":")
        if not separator or not module or not attribute:
            raise ValueError(f"invalid factory for {self.name}: {self.factory!r}")
        return getattr(importlib.import_module(module), attribute)


class Registry:
    def __init__(self, category: str):
        self.category = category
        self._entries: dict[str, Entry] = {}

    def register(self, entry: Entry) -> None:
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]*", entry.name):
            raise ValueError(f"invalid {self.category} name: {entry.name!r}")
        if entry.name in self._entries:
            raise ValueError(f"{self.category} already registered: {entry.name}")
        self._entries[entry.name] = entry

    def get(self, name: str) -> Entry:
        try:
            return self._entries[name]
        except KeyError as exc:
            raise ValueError(f"unknown {self.category}: {name!r}") from exc

    def entries(self) -> tuple[Entry, ...]:
        return tuple(self._entries[name] for name in sorted(self._entries))

    def create(self, name: str, *args: Any, **kwargs: Any):
        return self.get(name).load()(*args, **kwargs)


def policy_registry(policy_root: Path | None = None) -> Registry:
    registry = Registry("policy")
    registry.register(Entry("mock", "xpolicy.policy.mock:MockPolicy", ("chunk", "cpu")))
    if policy_root is None:
        import XPolicyLab

        policy_root = Path(XPolicyLab.__path__[0]) / "policy"
    if policy_root.is_dir():
        for directory in sorted(policy_root.iterdir()):
            if not directory.is_dir() or not directory.name.isidentifier():
                continue
            if not (directory / "model.py").is_file():
                continue
            capabilities = ("legacy", "chunk") if directory.name == "StarVLA" else ("legacy",)
            registry.register(Entry(
                directory.name, f"XPolicyLab.policy.{directory.name}.model:Model",
                capabilities, "XPolicyLab.policy",
            ))
    return registry
