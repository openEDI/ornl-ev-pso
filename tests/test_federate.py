"""Tests for EVCS federate configuration files and lifecycle."""

import json
import os
from importlib.util import find_spec

import pytest


def _evcs_base_dir():
    """Return the directory containing the evcs_federate package files."""
    spec = find_spec("evcs_federate.evcs_federate")
    return os.path.dirname(os.path.abspath(spec.origin))


def _repo_root():
    """Return the repository root (parent of tests/ directory)."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_config_json_valid():
    """config.json is valid JSON and contains required keys."""
    path = os.path.join(_evcs_base_dir(), "config.json")
    assert os.path.exists(path), f"config.json not found at {path}"
    with open(path) as f:
        data = json.load(f)
    assert "federate_config" in data, "config.json missing 'federate_config'"
    assert "simulation_config" in data, "config.json missing 'simulation_config'"
    fed_cfg = data["federate_config"]
    assert "name" in fed_cfg
    assert "coreType" in fed_cfg
    assert "port" in fed_cfg


def test_input_mapping_json_valid():
    """input_mapping.json is valid JSON and contains expected subscription keys."""
    path = os.path.join(_evcs_base_dir(), "input_mapping.json")
    assert os.path.exists(path), f"input_mapping.json not found at {path}"
    with open(path) as f:
        data = json.load(f)
    required_keys = {"powers_real_in", "powers_imag_in", "topology"}
    missing = required_keys - set(data.keys())
    assert not missing, f"input_mapping.json missing keys: {missing}"


def test_static_inputs_json_valid():
    """static_inputs.json is valid JSON and contains required EV parameters."""
    path = os.path.join(_evcs_base_dir(), "static_inputs.json")
    assert os.path.exists(path), f"static_inputs.json not found at {path}"
    with open(path) as f:
        data = json.load(f)
    required_keys = {
        "name",
        "evcs_bus",
        "evcs_bus_assignment",
        "num_evs_per_station",
        "max_charging_rate",
        "battery_capacity",
        "charging_efficiency",
        "desired_soc",
        "random_seed",
    }
    missing = required_keys - set(data.keys())
    assert not missing, f"static_inputs.json missing keys: {missing}"
    assert isinstance(data["evcs_bus"], list)
    assert isinstance(data["evcs_bus_assignment"], dict)
    assert isinstance(data["num_evs_per_station"], list)


def test_component_definition_json_valid():
    """component_definition.json is valid JSON and has expected port structure."""
    path = os.path.join(_repo_root(), "component_definition.json")
    assert os.path.exists(path), f"component_definition.json not found at {path}"
    with open(path) as f:
        data = json.load(f)
    assert "execute_function" in data, "component_definition.json missing 'execute_function'"
    assert "dynamic_inputs" in data, "component_definition.json missing 'dynamic_inputs'"
    assert "dynamic_outputs" in data, "component_definition.json missing 'dynamic_outputs'"

    input_port_ids = {p["port_id"] for p in data["dynamic_inputs"]}
    assert "powers_real_in" in input_port_ids
    assert "powers_imag_in" in input_port_ids
    assert "topology" in input_port_ids

    output_port_ids = {p["port_id"] for p in data["dynamic_outputs"]}
    assert "change_commands" in output_port_ids


def test_component_definition_no_directory_key():
    """component_definition.json must not contain a 'directory' key (non-standard)."""
    path = os.path.join(_repo_root(), "component_definition.json")
    with open(path) as f:
        data = json.load(f)
    assert "directory" not in data, (
        "component_definition.json must not have 'directory' key (single-component pattern)"
    )


def test_execute_function_format():
    """execute_function must use module path without 'src.' prefix."""
    path = os.path.join(_repo_root(), "component_definition.json")
    with open(path) as f:
        data = json.load(f)
    exec_fn = data["execute_function"]
    assert not exec_fn.startswith("python -m src."), (
        f"execute_function should not have 'src.' prefix, got: {exec_fn}"
    )
    assert "evcs_federate" in exec_fn, (
        f"execute_function should reference evcs_federate module, got: {exec_fn}"
    )


def test_federate_lifecycle():
    """Full HELICS federate lifecycle: setup, simulate, finalize.

    starts a local HELICS broker,
    runs finite timesteps (no real data needed - algorithm is skipped),
    then finalizes cleanly.
    """
    from oedisi.types.common import BrokerConfig

    from evcs_federate.evcs_federate import EVCSFederate

    base_dir = _evcs_base_dir()
    with open(os.path.join(base_dir, "input_mapping.json")) as f:
        input_mapping = json.load(f)

    broker_config = BrokerConfig(broker_ip="127.0.0.1", broker_port=23404)

    federate = EVCSFederate(
        federate_name="evcs_test",
        input_mapping=input_mapping,
        broker_config=broker_config,
        evcs_bus=["48.1"],
        test_mode=True,
    )
    federate.simulate(num_timesteps=4)
    federate.finalize()
