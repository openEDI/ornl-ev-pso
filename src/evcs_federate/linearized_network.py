"""fast voltage estimation during PSO."""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import spsolve

logger = logging.getLogger(__name__)


class LinearizedNetwork:
    """
    Voltage sensitivity matrix built from OEDISI Topology data.

    Uses LinDistFlow approximation: ΔV_j ≈ -(R·ΔP + X·ΔQ) / V₀
    Enables fast estimation V_new = V_base + (∂V/∂P) · ΔP without full power flow.
    """

    def __init__(
        self, bus_ids: List[str], base_voltages: np.ndarray, slack_bus: str = None
    ):
        """Initialize with bus list and per-unit base voltages."""
        self.bus_ids = list(bus_ids)
        self.n_buses = len(bus_ids)
        self.base_voltages = np.array(base_voltages)
        self.slack_bus = slack_bus

        self.bus_to_idx = {bus: i for i, bus in enumerate(bus_ids)}

        self.dV_dP = None  # ∂V/∂P sensitivity matrix
        self.dV_dQ = None  # ∂V/∂Q sensitivity matrix

        self.branches = []  # List of (from_bus, to_bus, R, X)

        logger.info(f"LinearizedNetwork initialized with {self.n_buses} buses")
        if slack_bus:
            logger.info(f"Slack bus: {slack_bus}")

    def add_branch(self, from_bus: str, to_bus: str, r_pu: float, x_pu: float):
        """Add a branch (line/transformer) with per-unit impedance."""
        if from_bus in self.bus_to_idx and to_bus in self.bus_to_idx:
            self.branches.append((from_bus, to_bus, r_pu, x_pu))
        else:
            logger.warning(f"Branch {from_bus}->{to_bus} skipped: bus not in network")

    def build_from_topology(self, topology):
        """Build network model from OEDISI Topology object received via HELICS."""
        logger.info("Building network from OEDISI Topology...")

        if hasattr(topology, "base_voltage_magnitudes"):
            self.bus_ids = list(topology.base_voltage_magnitudes.ids)
            self.base_voltages = np.array(topology.base_voltage_magnitudes.values)
            self.n_buses = len(self.bus_ids)
            self.bus_to_idx = {bus: i for i, bus in enumerate(self.bus_ids)}

        if hasattr(topology, "slack_bus") and topology.slack_bus:
            self.slack_bus = topology.slack_bus[0] if topology.slack_bus else None

        if hasattr(topology, "incidences"):
            from_buses = topology.incidences.from_equipment
            to_buses = topology.incidences.to_equipment

            # Use default impedances scaled by branch index (full R/X extraction not implemented)
            default_r = 0.01
            default_x = 0.03

            for i, (from_bus, to_bus) in enumerate(zip(from_buses, to_buses)):
                r_pu = default_r * (1 + 0.01 * i)
                x_pu = default_x * (1 + 0.01 * i)
                self.add_branch(str(from_bus), str(to_bus), r_pu, x_pu)

        logger.info(f"Built network with {len(self.branches)} branches")
        self.compute_sensitivity_matrix()

    def compute_sensitivity_matrix(self):
        """Compute ∂V/∂P and ∂V/∂Q via path-tracing from each bus to the slack."""
        logger.info("Computing voltage sensitivity matrix...")

        n = self.n_buses
        self.dV_dP = np.zeros((n, n))
        self.dV_dQ = np.zeros((n, n))

        # Build parent map assuming radial (tree) topology
        parent = {}
        for from_bus, to_bus, r, x in self.branches:
            if to_bus not in parent:
                parent[to_bus] = (from_bus, r, x)

        for j, bus_j in enumerate(self.bus_ids):
            path_r = 0.0
            path_x = 0.0
            current = bus_j

            while current in parent:
                parent_bus, r, x = parent[current]
                path_r += r
                path_x += x
                current = parent_bus

            # ΔV_j ≈ -ΔP * R_path / V₀  (simplified LinDistFlow, diagonal only)
            v0 = self.base_voltages[j] if j < len(self.base_voltages) else 1.0
            if v0 > 0:
                self.dV_dP[j, j] = -path_r / v0
                self.dV_dQ[j, j] = -path_x / v0

        logger.info(f"Sensitivity matrix computed: {n}x{n}")
        logger.debug(f"Sample dV/dP diagonal: {np.diag(self.dV_dP)[:5]}")

    def estimate_voltages(
        self, base_voltages: np.ndarray, ev_loads_per_bus: Dict[str, float]
    ) -> np.ndarray:
        """Estimate voltages using V_new = V_base + dV_dP @ delta_P (kW converted to per-unit MW)."""
        if self.dV_dP is None:
            logger.warning("Sensitivity matrix not computed, returning base voltages")
            return base_voltages.copy()

        delta_P = np.zeros(self.n_buses)
        for bus_id, load_kw in ev_loads_per_bus.items():
            if bus_id in self.bus_to_idx:
                idx = self.bus_to_idx[bus_id]
                delta_P[idx] = load_kw / 1000.0  # kW -> per-unit on 1 MVA base

        delta_V = self.dV_dP @ delta_P
        return base_voltages + delta_V

    def get_evcs_bus_indices(self, evcs_bus_ids: List[str]) -> List[int]:
        """Get matrix indices for a list of EVCS bus IDs."""
        indices = []
        for bus in evcs_bus_ids:
            if bus in self.bus_to_idx:
                indices.append(self.bus_to_idx[bus])
            else:
                logger.warning(f"EVCS bus {bus} not found in network")
        return indices

    def check_voltage_limits(
        self, voltages: np.ndarray, v_min: float = 0.95, v_max: float = 1.05
    ) -> bool:
        """Return True if all voltages are within [v_min, v_max]."""
        return np.all((voltages >= v_min) & (voltages <= v_max))

    def get_voltage_violations(
        self, voltages: np.ndarray, v_min: float = 0.95, v_max: float = 1.05
    ) -> Dict[str, float]:
        """Return dict of bus_id -> violation magnitude for all out-of-bounds buses."""
        violations = {}
        for i, (voltage, bus_id) in enumerate(zip(voltages, self.bus_ids)):
            if voltage < v_min:
                violations[bus_id] = v_min - voltage
            elif voltage > v_max:
                violations[bus_id] = voltage - v_max
        return violations


def create_simple_network(
    evcs_buses: List[str], num_buses: int = 10
) -> LinearizedNetwork:
    """Create a simple radial test network for development/testing."""
    bus_ids = [f"bus_{i}" for i in range(num_buses)]

    for evcs_bus in evcs_buses:
        if evcs_bus not in bus_ids:
            bus_ids.append(evcs_bus)

    base_voltages = np.ones(len(bus_ids))
    network = LinearizedNetwork(bus_ids, base_voltages, slack_bus="bus_0")

    for i in range(1, len(bus_ids)):
        network.add_branch(f"bus_{i-1}", bus_ids[i], r_pu=0.01, x_pu=0.03)

    network.compute_sensitivity_matrix()
    return network


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    evcs_buses = ["48.1", "65.1", "76.1"]
    network = create_simple_network(evcs_buses, num_buses=100)

    base_voltages = np.ones(len(network.bus_ids))
    ev_loads = {"48.1": 100.0, "65.1": 80.0, "76.1": 90.0}

    estimated = network.estimate_voltages(base_voltages, ev_loads)

    print(f"Base voltages (sample): {base_voltages[:5]}")
    print(f"Estimated voltages (sample): {estimated[:5]}")

    violations = network.get_voltage_violations(estimated)
    print(f"Violations: {violations}")
