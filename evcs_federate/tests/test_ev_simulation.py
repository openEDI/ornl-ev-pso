"""Tests for EV simulation"""
import numpy as np
from evcs_federate.ev_simulation import (
    uncontrolled_charging,
    calculate_soc,
    calculate_cost,
    simulate_real_charging_process,
)

def test_uncontrolled_charging():
    num_evs = 3
    num_steps = 24
    initial_soc = np.array([0.3, 0.2, 0.4])
    arrival_idx = np.array([7, 8, 9])
    departure_idx = np.array([20, 21, 22])

    soc, rate = uncontrolled_charging(
        initial_soc, num_steps, 1.0, 50.0, 0.95,
        arrival_idx, departure_idx, num_evs, 11.0, 1.0
    )
    assert soc.shape == (num_evs, num_steps)
    assert rate.shape == (num_evs, num_steps)
    # All EVs should charge (rate > 0 at some point)
    assert np.any(rate > 0)
    # Rate should be zero before arrival
    for ev in range(num_evs):
        assert np.all(rate[ev, :arrival_idx[ev]] == 0)

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
        initial_soc, scheduled, num_steps, 1.0, 50.0, 0.95,
        np.array([0]), np.array([10]), num_evs
    )
    # SOC should never exceed 1.0
    assert np.all(soc <= 1.0 + 1e-6)
