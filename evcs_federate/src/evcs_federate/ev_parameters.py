"""EV fleet parameter generation for multi-bus EVCS deployments."""

import numpy as np
import json
import os
from scipy.stats import truncnorm, lognorm

DEFAULT_CONFIG = {
    "num_evs_per_station": [15, 12, 13],  # EVs per station (total: 40)
    "max_charging_rate": 11,               # kW per charger
    "battery_capacity": 50,                # kWh per vehicle
    "charging_efficiency": 0.95,
    "desired_soc": 1.0,                    # Target SOC (100%)

    # Initial SOC: truncated normal distribution
    "soc_mean": 0.3,
    "soc_std": 0.2,
    "soc_lower": 0.1,
    "soc_upper": 0.5,

    # Arrival time: truncated normal (hours from midnight)
    "arrival_mean": 9.0,                   # 9 AM
    "arrival_std": 1.225,                  # sqrt(1.5)
    "arrival_lower": 7.0,                  # 7 AM
    "arrival_upper": 11.0,                 # 11 AM

    # Departure time: shifted lognormal
    "departure_shift": 17.5,               # 5:30 PM
    "departure_mu": 0.0,
    "departure_sigma": 0.9,

    "total_hours": 24,
    "control_interval": 0.25,             # 15 min = 0.25 hr
    "random_seed": 42,

    # Computed dynamically in generate_ev_parameters()
    "electricity_price": None
}


# Reference 24-hour price profile at 15-min resolution (96 values, $/kWh)
_PRICE_PROFILE_96 = [
    0.05753, 0.03334, 0.03098, 0.02518, 0.0319, 0.03044, 0.02022, 0.02363, 0.02296, 0.02188,
    0.02401, 0.02413, 0.02406, 0.03101, 0.03044, 0.02183, 0.03054, 0.02495, 0.05245, 0.0323,
    0.02488, 0.03324, 0.03119, 0.03099, 0.03202, 0.03339, 0.10425, 0.06402, 0.07178, 0.06245,
    0.04043, 0.04192, 0.06465, 0.0374, 0.03107, 0.0294, 0.03059, 0.02309, 0.0318, 0.02344,
    0.03023, 0.03506, 0.03684, 0.03301, 0.03361, 0.03681, 0.03325, 0.02933, 0.03206, 0.03128,
    0.02815, 0.02668, 0.02626, 0.02622, 0.02683, 0.02419, 0.02501, 0.02475, 0.02456, 0.02561,
    0.02566, 0.0253, 0.02475, 0.02508, 0.02603, 0.02533, 0.02698, 0.02886, 0.02569, 0.03306,
    0.03671, 0.03396, 0.03308, 0.03461, 0.04199, 0.03632, 0.03482, 0.10161, 0.09881, 0.11398,
    0.11103, 0.05463, 0.04681, 0.05261, 0.07763, 0.03797, 0.04209, 0.03728, 0.04444, 0.06747,
    0.03306, 0.03282, 0.02967, 0.03829, 0.03522, 0.03454
]


def generate_electricity_price(num_control_steps):
    """Resample the 96-point reference price profile to match num_control_steps."""
    ref = np.array(_PRICE_PROFILE_96)

    if num_control_steps == 96:
        return ref.copy()

    prices = np.zeros(num_control_steps)
    for i in range(num_control_steps):
        start_frac = i / num_control_steps
        end_frac = (i + 1) / num_control_steps
        start_idx = int(start_frac * 96)
        end_idx = max(start_idx + 1, int(end_frac * 96))
        end_idx = min(end_idx, 96)
        prices[i] = np.mean(ref[start_idx:end_idx])

    return prices


def load_config(config_path="static_inputs.json"):
    """Load configuration from file, returning empty dict if not found."""
    try:
        with open(config_path) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def generate_ev_parameters(config=None):
    """Generate all EV fleet parameters from config, falling back to DEFAULT_CONFIG."""
    if config is None:
        config = load_config()

    cfg = {**DEFAULT_CONFIG, **config}

    np.random.seed(cfg.get("random_seed", 42))

    num_evs_per_station = cfg["num_evs_per_station"]
    num_evs = sum(num_evs_per_station)
    control_interval = cfg["control_interval"]
    total_hours = cfg["total_hours"]
    num_control_steps = int(total_hours / control_interval)

    soc_mean = cfg["soc_mean"]
    soc_std = cfg["soc_std"]
    soc_lower = cfg["soc_lower"]
    soc_upper = cfg["soc_upper"]

    arr_mean = cfg["arrival_mean"]
    arr_std = cfg["arrival_std"]
    arr_lower = cfg["arrival_lower"]
    arr_upper = cfg["arrival_upper"]

    dep_shift = cfg["departure_shift"]
    dep_mu = cfg["departure_mu"]
    dep_sigma = cfg["departure_sigma"]

    battery_capacity = cfg["battery_capacity"]
    desired_soc = cfg["desired_soc"]

    arrival_time_idx_stations = []
    departure_time_idx_stations = []
    initial_soc_stations = []
    charging_energy_stations = []

    ev_offset = 0
    evcs_bus = cfg.get("evcs_bus", ["48.1", "65.1", "76.1"])

    user_assignment = cfg.get("evcs_bus_assignment")
    if user_assignment and len(user_assignment) > 0:
        evcs_bus_assignment = user_assignment
    else:
        evcs_bus_assignment = {}
        for station_idx, n_evs in enumerate(num_evs_per_station):
            if station_idx < len(evcs_bus):
                bus_id = evcs_bus[station_idx]
                evcs_bus_assignment[bus_id] = list(range(ev_offset, ev_offset + n_evs))
            ev_offset += n_evs
        ev_offset = 0

    for station_idx, n_evs in enumerate(num_evs_per_station):
        # Arrival: truncated normal
        a_arr = (arr_lower - arr_mean) / arr_std
        b_arr = (arr_upper - arr_mean) / arr_std
        arrival_hr = truncnorm.rvs(a_arr, b_arr, loc=arr_mean, scale=arr_std, size=n_evs)
        arrival_idx = np.floor(arrival_hr / control_interval).astype(int)
        arrival_idx = np.clip(arrival_idx, 0, num_control_steps - 1)

        # Departure: shifted lognormal, constrained within simulation window and after arrival
        departure_hr = np.zeros(n_evs)
        departure_idx = np.zeros(n_evs, dtype=int)

        for i in range(n_evs):
            max_attempts = 1000
            attempts = 0
            while attempts < max_attempts:
                attempts += 1
                x = lognorm.rvs(s=dep_sigma, scale=np.exp(dep_mu))
                t_dep = x + dep_shift

                if t_dep <= total_hours and t_dep > arrival_hr[i]:
                    dep_idx = int(np.floor(t_dep / control_interval))
                    if dep_idx > arrival_idx[i]:
                        departure_hr[i] = t_dep
                        departure_idx[i] = min(dep_idx, num_control_steps)
                        break

            if attempts >= max_attempts:
                departure_idx[i] = min(arrival_idx[i] + 4, num_control_steps)

        # Initial SOC: truncated normal
        a_soc = (soc_lower - soc_mean) / soc_std
        b_soc = (soc_upper - soc_mean) / soc_std
        initial_soc_station = truncnorm.rvs(a_soc, b_soc, loc=soc_mean, scale=soc_std, size=n_evs)
        initial_soc_station = np.clip(initial_soc_station, soc_lower, soc_upper)

        charging_energy_station = (desired_soc - initial_soc_station) * battery_capacity

        arrival_time_idx_stations.append(arrival_idx)
        departure_time_idx_stations.append(departure_idx)
        initial_soc_stations.append(initial_soc_station)
        charging_energy_stations.append(charging_energy_station)

        ev_offset += n_evs

    if cfg.get("electricity_price") is not None:
        electricity_price = np.array(cfg["electricity_price"])
        if len(electricity_price) != num_control_steps:
            electricity_price = generate_electricity_price(num_control_steps)
    else:
        electricity_price = generate_electricity_price(num_control_steps)

    return {
        "num_evs": num_evs,
        "num_evs_per_station": num_evs_per_station,
        "max_charging_rate": cfg["max_charging_rate"],
        "battery_capacity": battery_capacity,
        "charging_efficiency": cfg["charging_efficiency"],
        "desired_soc": desired_soc,
        "control_interval": control_interval,
        "total_hours": total_hours,
        "num_control_steps": num_control_steps,
        "electricity_price": electricity_price,
        "arrival_time_idx": np.concatenate(arrival_time_idx_stations),
        "departure_time_idx": np.concatenate(departure_time_idx_stations),
        "initial_soc": np.concatenate(initial_soc_stations),
        "charging_energy": np.concatenate(charging_energy_stations),
        "arrival_time_idx_stations": arrival_time_idx_stations,
        "departure_time_idx_stations": departure_time_idx_stations,
        "initial_soc_stations": initial_soc_stations,
        "charging_energy_stations": charging_energy_stations,
        "evcs_bus_assignment": evcs_bus_assignment,
    }


# Backward-compatible global variables generated from DEFAULT_CONFIG at import time
_default_params = generate_ev_parameters(DEFAULT_CONFIG)

num_evs = _default_params["num_evs"]
max_charging_rate = _default_params["max_charging_rate"]
charging_efficiency = _default_params["charging_efficiency"]
battery_capacity = _default_params["battery_capacity"]
control_interval = _default_params["control_interval"]
total_hours = _default_params["total_hours"]
num_control_steps = _default_params["num_control_steps"]
electricity_price = _default_params["electricity_price"]
arrival_time_idx = _default_params["arrival_time_idx"]
departure_time_idx = _default_params["departure_time_idx"]
initial_soc = _default_params["initial_soc"]
charging_energy = _default_params["charging_energy"]
desired_state_of_charge = _default_params["desired_soc"]
evcs_bus_assignment = _default_params["evcs_bus_assignment"]

arrival_time_hr = None
departure_time_hr = None
t0_dep = DEFAULT_CONFIG["departure_shift"]
mu_a_dep = DEFAULT_CONFIG["departure_mu"]
sigma_a_dep = DEFAULT_CONFIG["departure_sigma"]
dep_upper_bound_hr = DEFAULT_CONFIG["total_hours"]
soc_ini_mean = DEFAULT_CONFIG["soc_mean"]
soc_ini_std_dev = DEFAULT_CONFIG["soc_std"]
soc_ini_lower_bound = DEFAULT_CONFIG["soc_lower"]
soc_ini_upper_bound = DEFAULT_CONFIG["soc_upper"]


def get_bus_for_ev(ev_index, bus_assignment=None):
    """Return the bus ID where the given EV is located."""
    if bus_assignment is None:
        bus_assignment = evcs_bus_assignment

    for bus, ev_list in bus_assignment.items():
        if ev_index in ev_list:
            return bus
    return list(bus_assignment.keys())[0]


def get_evs_at_bus(bus_id, bus_assignment=None):
    """Return list of EV indices assigned to a specific bus."""
    if bus_assignment is None:
        bus_assignment = evcs_bus_assignment

    return bus_assignment.get(bus_id, [])


def update_global_parameters(config):
    """Regenerate and update all global EV parameters from a new config dict."""
    global num_evs, max_charging_rate, charging_efficiency, battery_capacity
    global control_interval, total_hours, num_control_steps, electricity_price
    global arrival_time_idx, departure_time_idx, initial_soc, charging_energy
    global desired_state_of_charge, evcs_bus_assignment

    params = generate_ev_parameters(config)

    num_evs = params["num_evs"]
    max_charging_rate = params["max_charging_rate"]
    charging_efficiency = params["charging_efficiency"]
    battery_capacity = params["battery_capacity"]
    control_interval = params["control_interval"]
    total_hours = params["total_hours"]
    num_control_steps = params["num_control_steps"]
    electricity_price = params["electricity_price"]
    arrival_time_idx = params["arrival_time_idx"]
    departure_time_idx = params["departure_time_idx"]
    initial_soc = params["initial_soc"]
    charging_energy = params["charging_energy"]
    desired_state_of_charge = params["desired_soc"]
    evcs_bus_assignment = params["evcs_bus_assignment"]
