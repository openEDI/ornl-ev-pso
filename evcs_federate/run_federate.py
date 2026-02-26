"""Launcher for oedisi run compatibility with src-layout."""
import sys
import os

# Add src directory to Python path so package imports work
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from evcs_federate.evcs_federate import run_simulator  # noqa: E402
from oedisi.types.common import BrokerConfig  # noqa: E402

if __name__ == "__main__":
    run_simulator(BrokerConfig(broker_ip="127.0.0.1"))
