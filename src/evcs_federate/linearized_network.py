"""Fast voltage estimation during PSO."""

import logging
from typing import Dict, List

import numpy as np

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

    def _assemble_admittance(self, admittance):
        if hasattr(admittance, "admittance_matrix") and hasattr(admittance, "ids"):
            ids = list(admittance.ids)
            mat = admittance.admittance_matrix
            n = len(ids)
            Y = np.zeros((n, n), dtype=complex)
            for i in range(n):
                row = mat[i]
                for j in range(n):
                    re, im = row[j]
                    Y[i, j] = complex(re, im)
            return ids, Y

        if hasattr(admittance, "from_equipment") and hasattr(
            admittance, "admittance_list"
        ):
            from_eq = list(admittance.from_equipment)
            to_eq = list(admittance.to_equipment)
            vals = list(admittance.admittance_list)
            ids = sorted(set(from_eq) | set(to_eq))
            pos = {bid: k for k, bid in enumerate(ids)}
            n = len(ids)
            Y = np.zeros((n, n), dtype=complex)
            for f, t, v in zip(from_eq, to_eq, vals):
                re, im = v
                Y[pos[f], pos[t]] += complex(re, im)
            return ids, Y

        raise ValueError("Unrecognized admittance format")

    def build_sensitivity_from_admittance(
        self, topology, s_base: float = 1e6, diagonal_only: bool = False,
        sensitivity_scale: float = 1.0
    ):
        admittance = topology.admittance
        bvm = topology.base_voltage_magnitudes
        base_v_by_id = dict(zip(bvm.ids, bvm.values))

        adm_ids, Y = self._assemble_admittance(admittance)
        n_adm = len(adm_ids)
        if n_adm == 0:
            raise ValueError("Empty admittance matrix")

        base_v = np.array(
            [base_v_by_id.get(bid, 0.0) for bid in adm_ids], dtype=float
        )
        if np.any(base_v <= 0.0):
            raise ValueError("Missing base voltage for one or more admittance ids")

        Y_pu = Y * (np.outer(base_v, base_v) / float(s_base))

        slack_ids = set(topology.slack_bus) if topology.slack_bus else set()
        keep = [i for i, bid in enumerate(adm_ids) if bid not in slack_ids]
        if len(keep) < 2:
            raise ValueError("Not enough non-slack buses in admittance")

        Y_red = Y_pu[np.ix_(keep, keep)]
        Z_red = np.linalg.inv(Y_red)
        dVdP_red = -np.real(Z_red)
        dVdQ_red = -np.imag(Z_red)

        red_pos = {adm_ids[keep[k]]: k for k in range(len(keep))}

        n = self.n_buses
        full_dVdP = np.zeros((n, n))
        full_dVdQ = np.zeros((n, n))
        for i, bus_i in enumerate(self.bus_ids):
            pi = red_pos.get(bus_i)
            if pi is None:
                continue
            for j, bus_j in enumerate(self.bus_ids):
                pj = red_pos.get(bus_j)
                if pj is None:
                    continue
                full_dVdP[i, j] = dVdP_red[pi, pj]
                full_dVdQ[i, j] = dVdQ_red[pi, pj]

        if sensitivity_scale != 1.0:
            full_dVdP = full_dVdP * sensitivity_scale
            full_dVdQ = full_dVdQ * sensitivity_scale
        if diagonal_only:
            full_dVdP = np.diag(np.diag(full_dVdP))
            full_dVdQ = np.diag(np.diag(full_dVdQ))
        self.dV_dP = full_dVdP
        self.dV_dQ = full_dVdQ
        logger.info(
            f"Built Zbus dV/dP sensitivity from admittance: {n}x{n} "
            f"({len(keep)} non-slack of {n_adm} admittance nodes, "
            f"diagonal_only={diagonal_only})"
        )

    def estimate_voltages(
        self, base_voltages: np.ndarray, ev_loads_per_bus: Dict[str, float]
    ) -> np.ndarray:
        """Estimate voltages using V_new = V_base + dV_dP @ delta_P (kW converted to per-unit MW)."""
        if self.dV_dP is None:
            logger.warning("Sensitivity matrix not computed, returning base voltages")
            return base_voltages.copy()

        delta_P = np.zeros(self.n_buses)
        if getattr(self, "_lower_idx_n", -1) != len(self.bus_to_idx):
            self._lower_idx = {str(k).lower(): v for k, v in self.bus_to_idx.items()}
            self._lower_idx_n = len(self.bus_to_idx)
        for bus_id, load_kw in ev_loads_per_bus.items():
            idx = self._lower_idx.get(str(bus_id).lower())
            if idx is not None:
                delta_P[idx] = load_kw / 1000.0

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

    logger.info(f"Base voltages (sample): {base_voltages[:5]}")
    logger.info(f"Estimated voltages (sample): {estimated[:5]}")

    violations = network.get_voltage_violations(estimated)
    logger.info(f"Violations: {violations}")
