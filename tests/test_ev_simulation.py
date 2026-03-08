"""Tests for EV simulation"""

import json

import numpy as np
from oedisi.types.data_types import Command, CommandList

from evcs_federate.ev_simulation import (
    calculate_cost,
    calculate_soc,
    simulate_real_charging_process,
    uncontrolled_charging,
)
from evcs_federate.evcs_federate import build_change_commands, bus_id_to_load_name


def test_uncontrolled_charging():
    num_evs = 3
    num_steps = 24
    initial_soc = np.array([0.3, 0.2, 0.4])
    arrival_idx = np.array([7, 8, 9])
    departure_idx = np.array([20, 21, 22])

    soc, rate = uncontrolled_charging(
        initial_soc,
        num_steps,
        1.0,
        50.0,
        0.95,
        arrival_idx,
        departure_idx,
        num_evs,
        11.0,
        1.0,
    )
    assert soc.shape == (num_evs, num_steps)
    assert rate.shape == (num_evs, num_steps)
    # All EVs should charge (rate > 0 at some point)
    assert np.any(rate > 0)
    # Rate should be zero before arrival
    for ev in range(num_evs):
        assert np.all(rate[ev, : arrival_idx[ev]] == 0)


def test_calculate_cost():
    rate = np.array([[10.0, 5.0, 0.0]])
    price = np.array([0.05, 0.10, 0.03])
    cost = calculate_cost(rate, price, 3, 1.0, 1, np.array([0]), np.array([3]))
    assert cost > 0
    expected = 10.0 * 1.0 * 0.05 + 5.0 * 1.0 * 0.10
    assert abs(cost - expected) < 0.01


def test_simulate_real_charging_no_overcharge():
    num_evs = 1
    num_steps = 10
    initial_soc = np.array([0.9])
    scheduled = np.zeros((1, 10))
    scheduled[0, 0:5] = 50.0

    soc, real_rate = simulate_real_charging_process(
        initial_soc,
        scheduled,
        num_steps,
        1.0,
        50.0,
        0.95,
        np.array([0]),
        np.array([10]),
        num_evs,
    )
    # SOC should never exceed 1.0
    assert np.all(soc <= 1.0 + 1e-6)


def test_bus_id_to_load_name():
    # 3-phase buses (47, 48) use S{num} without phase suffix
    assert bus_id_to_load_name("48.1") == "S48"
    assert bus_id_to_load_name("47.1") == "S47"
    # Single-phase buses use S{num}{phase_letter}
    assert bus_id_to_load_name("65.1") == "S65a"
    assert bus_id_to_load_name("65.2") == "S65b"
    assert bus_id_to_load_name("76.3") == "S76c"


def test_build_change_commands_additive():
    """With base load, kW should be base + ev."""
    buses = ["48.1", "65.1"]
    ev_load = [50.0, 30.0]
    base_kw = {"48.1": 70.0, "65.1": 35.0}

    cmd_list = build_change_commands(buses, ev_load, base_kw)

    assert len(cmd_list.root) == 4  # 2 commands per bus
    # Bus 48.1: base=70 + ev=50 = 120
    assert cmd_list.root[0].obj_name == "Load.S48"
    assert cmd_list.root[0].obj_property == "kW"
    assert cmd_list.root[0].val == "120.0"
    assert cmd_list.root[1].obj_property == "kvar"
    # Bus 65.1: base=35 + ev=30 = 65
    assert cmd_list.root[2].obj_name == "Load.S65a"
    assert cmd_list.root[2].val == "65.0"

    # Serializes to valid JSON
    parsed = json.loads(cmd_list.model_dump_json())
    assert isinstance(parsed, list)
    assert len(parsed) == 4


def test_build_change_commands_zero_ev():
    """With zero EV load, kW should equal base load."""
    buses = ["65.1"]
    ev_load = [0.0]
    base_kw = {"65.1": 35.0}

    cmd_list = build_change_commands(buses, ev_load, base_kw)
    assert cmd_list.root[0].val == "35.0"


def test_build_change_commands_no_base():
    """Without base load info, kW should be just EV power."""
    buses = ["76.1"]
    ev_load = [50.0]

    cmd_list = build_change_commands(buses, ev_load)
    assert cmd_list.root[0].obj_name == "Load.S76a"
    assert cmd_list.root[0].val == "50.0"


def test_build_change_commands_serialization():
    """CommandList serializes to valid JSON array."""
    buses = ["48.1", "65.1", "76.1"]
    ev_load = [50.0, 30.0, 0.0]
    base_kw = {"48.1": 70.0, "65.1": 35.0, "76.1": 105.0}

    cmd_list = build_change_commands(buses, ev_load, base_kw)
    json_str = cmd_list.model_dump_json()
    parsed = json.loads(json_str)
    assert isinstance(parsed, list)
    assert len(parsed) == 6  # 2 per bus (kW + kvar)
