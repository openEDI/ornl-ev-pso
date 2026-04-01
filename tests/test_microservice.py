"""Tests for the EVCS federate microservice API.
- Subprocess fixture to start server
- Tests the full endpoint lifecycle: health -> configure -> run
"""

import os
import shlex
import signal
import subprocess
import sys
import time

import pytest
import requests

PORT = "5683"
URL = f"http://127.0.0.1:{PORT}"


@pytest.fixture
def init():
    """Start the EVCS server as a subprocess."""
    env = {**os.environ, "PORT": PORT}

    if sys.platform == "win32":
        proc = subprocess.Popen(
            [sys.executable, "-m", "evcs_federate.server"],
            env=env,
        )
    else:
        proc = subprocess.Popen(
            shlex.split("python3 -m evcs_federate.server"),
            env=env,
            preexec_fn=os.setpgrp,
        )

    time.sleep(3)
    yield
    if sys.platform == "win32":
        proc.terminate()
        proc.wait(timeout=10)
    else:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)


def test_health_check(init):
    """GET / returns 200 with hostname and host_ip."""
    res = requests.get(URL + "/")
    assert res.status_code == 200
    data = res.json()
    assert "hostname" in data
    assert "host_ip" in data


def test_configure(init):
    """POST /configure writes static_inputs.json and input_mapping.json."""
    payload = {
        "component": {
            "name": "evcs_test",
            "type": "EVCS",
            "parameters": {
                "evcs_bus": ["48.1", "65.1"],
                "num_evs_per_station": [5, 5],
                "max_charging_rate": 11.0,
                "battery_capacity": 50.0,
                "charging_efficiency": 0.95,
                "desired_soc": 0.9,
                "total_hours": 24,
                "control_interval": 0.25,
                "random_seed": 42,
            },
        },
        "links": [
            {
                "source": "feeder",
                "source_port": "powers_real",
                "target": "evcs_test",
                "target_port": "powers_real_in",
            },
            {
                "source": "feeder",
                "source_port": "powers_imag",
                "target": "evcs_test",
                "target_port": "powers_imag_in",
            },
            {
                "source": "feeder",
                "source_port": "topology",
                "target": "evcs_test",
                "target_port": "topology",
            },
        ],
    }
    res = requests.post(URL + "/configure", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "detail" in data


def test_run_endpoint(init):
    """POST /run with BrokerConfig returns 200."""
    payload = {
        "broker_ip": "127.0.0.1",
        "broker_port": 23404,
    }
    res = requests.post(URL + "/run", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "detail" in data
