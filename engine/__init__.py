"""Simulation engine for the hcu-train-web UI.

Everything that touches numbers delegates to `hcu_train_simulator` in-process.
"""

from . import config_builder, engine, hf_client, net, paths  # noqa: F401

__all__ = ["config_builder", "engine", "hf_client", "net", "paths"]