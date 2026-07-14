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
from evcs_federate.evcs_federate import build_change_commands


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
    assert np.any(rate > 0)
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
    assert np.all(soc <= 1.0 + 1e-6)


def test_build_change_commands_sets_ev_load():
    """Each bus gets a kW command with its EV load and a kvar=0 command."""
    buses = ["48.1", "65.1"]
    ev_load = [50.0, 30.0]

    cmd_list = build_change_commands(buses, ev_load)

    assert len(cmd_list.root) == 4  # kW + kvar per bus
    assert cmd_list.root[0].obj_name == "EVLoad.48.1"
    assert cmd_list.root[0].obj_property == "kW"
    assert cmd_list.root[0].val == "50.0"
    assert cmd_list.root[1].obj_name == "EVLoad.48.1"
    assert cmd_list.root[1].obj_property == "kvar"
    assert cmd_list.root[1].val == "0.0"
    assert cmd_list.root[2].obj_name == "EVLoad.65.1"
    assert cmd_list.root[2].val == "30.0"


def test_build_change_commands_zero_ev():
    """Zero EV load still emits a kW=0 command for that bus."""
    cmd_list = build_change_commands(["65.1"], [0.0])
    assert cmd_list.root[0].obj_property == "kW"
    assert cmd_list.root[0].val == "0.0"


def test_build_change_commands_serialization():
    """CommandList serializes to a valid JSON array, 2 commands per bus."""
    buses = ["48.1", "65.1", "76.1"]
    ev_load = [50.0, 30.0, 0.0]

    cmd_list = build_change_commands(buses, ev_load)
    parsed = json.loads(cmd_list.model_dump_json())
    assert isinstance(parsed, list)
    assert len(parsed) == 6
