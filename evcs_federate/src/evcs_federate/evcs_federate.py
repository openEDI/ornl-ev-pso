"""EVCS Federate - subscribes to topology and voltages, runs PSO with voltage estimation."""

import logging
import helics as h
import json
from datetime import datetime
from oedisi.types.common import BrokerConfig
from oedisi.types.data_types import (
    PowersImaginary,
    PowersReal,
    VoltagesReal,
    VoltagesImaginary,
    Topology,
)
from . import ev_simulation
from .ev_parameters import generate_ev_parameters
from .linearized_network import LinearizedNetwork
import numpy as np

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

# 3-phase buses in gadal_ieee123 use S{num} load names (no phase suffix)
THREE_PHASE_BUSES = {'47', '48'}


def bus_id_to_load_name(bus_id):
    """Convert bus ID (e.g., '48.1') to OpenDSS load name."""
    phase_map = {'1': 'a', '2': 'b', '3': 'c'}
    try:
        parts = bus_id.split('.')
        if len(parts) == 2:
            bus_num = parts[0]
            phase_num = parts[1]
            if bus_num in THREE_PHASE_BUSES:
                return f"S{bus_num}"
            phase_letter = phase_map.get(phase_num, 'a')
            return f"S{bus_num}{phase_letter}"
    except Exception:
        pass
    return bus_id


class EVCSFederate:

    def __init__(
        self,
        federate_name,
        input_mapping,
        broker_config: BrokerConfig,
        evcs_bus: list = None,
        ev_params: dict = None,
        control_mode: str = "dopf",
    ):
        "Initializes federate with name and remaps input into subscriptions"
        self.evcs_bus = evcs_bus if evcs_bus is not None else ["48.1"]
        self.control_mode = control_mode
        logger.info(f"EVCS bus location(s): {self.evcs_bus}")
        logger.info(f"Control mode: {self.control_mode}")

        self.ev_params = ev_params
        if ev_params is not None:
            logger.info(f"Loaded EV config: {ev_params['num_evs']} EVs, "
                       f"{ev_params['num_control_steps']} control steps")

        deltat = 1

        fedinfo = h.helicsCreateFederateInfo()
        h.helicsFederateInfoSetBroker(fedinfo, broker_config.broker_ip)
        h.helicsFederateInfoSetBrokerPort(fedinfo, broker_config.broker_port)
        fedinfo.core_name = federate_name
        fedinfo.core_type = h.HELICS_CORE_TYPE_ZMQ
        fedinfo.core_init = "--federates=1"
        h.helicsFederateInfoSetTimeProperty(
            fedinfo, h.helics_property_time_delta, deltat
        )

        self.vfed = h.helicsCreateValueFederate(federate_name, fedinfo)
        logger.info("Value federate created")

        self.sub_power_P = self.vfed.register_subscription(
            input_mapping["powers_real_in"], "W"
        )
        self.sub_power_Q = self.vfed.register_subscription(
            input_mapping["powers_imag_in"], "W"
        )
        self.sub_topology = self.vfed.register_subscription(
            input_mapping["topology"], ""
        )
        self.sub_voltages_real = self.vfed.register_subscription(
            input_mapping["voltages_real"], ""
        )
        self.sub_voltages_imag = self.vfed.register_subscription(
            input_mapping["voltages_imag"], ""
        )
        logger.info("Subscribed to topology and voltages for linearized network")

        self.pub_ev_load_real = self.vfed.register_publication(
            "ev_load_real", h.HELICS_DATA_TYPE_STRING, ""
        )
        self.pub_ev_load_imag = self.vfed.register_publication(
            "ev_load_imag", h.HELICS_DATA_TYPE_STRING, ""
        )

        self.network = None

    def run(self):
        """Main run loop: build LinearizedNetwork from topology, then run PSO each timestep."""
        logger.info(f"Federate connected: {datetime.now()}")
        logger.info("=" * 60)
        logger.info("EVCS FEDERATE")
        logger.info(f"  Target buses: {self.evcs_bus}")

        evcs_bus_assignment = (self.ev_params.get("evcs_bus_assignment", {})
                               if self.ev_params else {})
        for bus, evs in evcs_bus_assignment.items():
            if evs:
                logger.info(f"    Bus {bus}: {len(evs)} EVs (indices {evs[0]}-{evs[-1]})")
        logger.info("=" * 60)

        self.vfed.enter_executing_mode()
        granted_time = h.helicsFederateRequestTime(self.vfed, h.HELICS_TIME_MAXTIME)

        num_particles = 30
        max_iterations = 30
        timestep_count = 0
        network_built = False

        import time as time_module

        while granted_time < h.HELICS_TIME_MAXTIME:
            if not self.sub_power_P.is_updated():
                granted_time = h.helicsFederateRequestTime(
                    self.vfed, h.HELICS_TIME_MAXTIME
                )
                continue

            timestep_count += 1
            logger.info("")
            logger.info("=" * 60)
            logger.info(f"TIMESTEP {timestep_count} | HELICS Time: {granted_time}")
            logger.info("=" * 60)

            if not network_built and self.sub_topology.is_updated():
                try:
                    topology = Topology.parse_obj(self.sub_topology.json)
                    logger.info("Received topology from feeder")

                    # IEEE 123-bus base voltage is 2400V L-N
                    V_BASE = 2400.0
                    base_volt_raw = np.array(topology.base_voltage_magnitudes.values)
                    base_volt_pu = base_volt_raw / V_BASE

                    self.network = LinearizedNetwork(
                        bus_ids=list(topology.base_voltage_magnitudes.ids),
                        base_voltages=base_volt_pu,
                        slack_bus=topology.slack_bus[0] if topology.slack_bus else None
                    )
                    self.network.build_from_topology(topology)
                    network_built = True
                    logger.info(f"Built LinearizedNetwork with {self.network.n_buses} buses")
                except Exception as e:
                    logger.warning(f"Could not build network from topology: {e}")
                    self.network = None

            base_voltages = None
            if self.sub_voltages_real.is_updated():
                try:
                    voltages_real = VoltagesReal.parse_obj(self.sub_voltages_real.json)
                    # IEEE 123-bus base voltage is 2400V L-N
                    V_BASE = 2400.0
                    voltages_raw = np.array(voltages_real.values)
                    base_voltages = voltages_raw / V_BASE
                    logger.info(f"Received {len(base_voltages)} voltages from feeder (converted to pu)")
                except Exception as e:
                    logger.warning(f"Could not parse voltages: {e}")

            if base_voltages is None:
                if self.network is not None:
                    base_voltages = np.ones(self.network.n_buses)
                else:
                    base_voltages = np.ones(100)

            power_P = PowersReal.parse_obj(self.sub_power_P.json)
            power_Q = PowersImaginary.parse_obj(self.sub_power_Q.json)

            load_ids = list(power_P.ids)
            logger.info(f"[INPUT] Received feeder data: {len(load_ids)} load buses")

            time = power_P.time
            time_idx = int(granted_time)

            pso_start = time_module.time()

            if self.control_mode == "uncontrolled":
                if not hasattr(self, '_uncontrolled_rate'):
                    ep = self.ev_params
                    _, self._uncontrolled_rate = ev_simulation.uncontrolled_charging(
                        ep["initial_soc"], ep["num_control_steps"], ep["control_interval"],
                        ep["battery_capacity"], ep["charging_efficiency"],
                        ep["arrival_time_idx"], ep["departure_time_idx"],
                        ep["num_evs"], ep["max_charging_rate"], ep["desired_soc"]
                    )
                    logger.info(f"[UNCONTROLLED] Computed greedy baseline schedule")
                charging_rate = self._uncontrolled_rate
                true_cost = 0.0
                logger.info(f"[UNCONTROLLED] Using greedy charging (no PSO)")
            else:
                logger.info(f"[PSO] Starting optimization: {num_particles} particles, {max_iterations} iterations")
                charging_rate, true_cost = ev_simulation.ev_pso_optimization(
                    num_particles, max_iterations, self.network, base_voltages,
                    self.evcs_bus, ev_params=self.ev_params
                )
                logger.info(f"[PSO] True electricity cost: ${true_cost:.2f}")

            pso_duration = time_module.time() - pso_start
            logger.info(f"[OPT] Complete in {pso_duration:.2f} seconds")

            ev_load_values = []
            ev_load_per_bus = {}
            for bus in self.evcs_bus:
                ev_indices = evcs_bus_assignment.get(bus, [])
                if ev_indices:
                    bus_power = float(np.sum(charging_rate[ev_indices, time_idx]))
                    num_charging = int(np.sum(charging_rate[ev_indices, time_idx] > 0))
                else:
                    bus_power = 0.0
                    num_charging = 0
                ev_load_values.append(bus_power)
                ev_load_per_bus[bus] = bus_power
                logger.info(f"[RESULT] Bus {bus}: {num_charging} EVs charging, Power: {bus_power:.2f} kW")

            total_ev_load = sum(ev_load_values)
            logger.info(f"[RESULT] Total across all buses: {total_ev_load:.2f} kW")

            load_names = [bus_id_to_load_name(bus) for bus in self.evcs_bus]
            equipment_ids = [bus.split('.')[0] for bus in self.evcs_bus]

            ev_load_real = PowersReal(
                ids=load_names,
                equipment_ids=equipment_ids,
                values=ev_load_values,
                time=time
            )
            ev_load_imag = PowersImaginary(
                ids=load_names,
                equipment_ids=equipment_ids,
                values=[0.0] * len(self.evcs_bus),
                time=time
            )

            self.pub_ev_load_real.publish(ev_load_real.json())
            self.pub_ev_load_imag.publish(ev_load_imag.json())
            logger.info(f"[OUTPUT] Published EV loads: {ev_load_per_bus}")
            logger.info("-" * 60)

            granted_time = h.helicsFederateRequestTime(self.vfed, h.HELICS_TIME_MAXTIME)

        self.stop()

    def stop(self):
        h.helicsFederateDisconnect(self.vfed)
        h.helicsFederateFree(self.vfed)
        h.helicsCloseLibrary()
        logger.info(f"Federate disconnected: {datetime.now()}")


def run_simulator(broker_config: BrokerConfig):
    logger.info(f"Running---------------------------------------------------")
    with open("static_inputs.json") as f:
        config = json.load(f)
        federate_name = config["name"]
        evcs_bus = config.get("evcs_bus", ["48.1"])
        control_mode = config.get("control_mode", "dopf")
        logger.info(f"Loaded evcs_bus from config: {evcs_bus}")
        logger.info(f"Control mode from config: {control_mode}")

    ev_params = generate_ev_parameters(config)
    logger.info(f"Generated EV parameters: {ev_params['num_evs']} EVs across "
                f"{len(ev_params['num_evs_per_station'])} stations")

    with open("input_mapping.json") as f:
        input_mapping = json.load(f)

    try:
        sfed = EVCSFederate(
            federate_name, input_mapping, broker_config, evcs_bus,
            ev_params=ev_params, control_mode=control_mode
        )
        logger.info("Value federate created")
    except h.HelicsException as e:
        logger.error(f"Failed to create HELICS Value Federate: {str(e)}")
        return

    sfed.run()
    logger.info(f"Running------------------------------------------------")

if __name__ == "__main__":
    run_simulator(BrokerConfig(broker_ip="127.0.0.1"))
