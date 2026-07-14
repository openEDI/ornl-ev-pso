"""EVCS Federate - subscribes to topology and voltages"""

import json
import logging
import time as time_module
from datetime import datetime

import helics as h
import numpy as np
from oedisi.types.common import BrokerConfig
from oedisi.types.data_types import (
    Command,
    CommandList,
    PowersImaginary,
    PowersReal,
    Topology,
    VoltagesImaginary,
    VoltagesReal,
)

from . import ev_simulation
from .ev_parameters import generate_ev_parameters
from .linearized_network import LinearizedNetwork

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

def build_change_commands(evcs_buses, ev_load_values):
    commands = []
    for bus, ev_kw in zip(evcs_buses, ev_load_values):
        commands.append(
            Command(
                obj_name=f"EVLoad.{bus}",
                obj_property="kW",
                val=str(ev_kw),
            )
        )
        commands.append(
            Command(
                obj_name=f"EVLoad.{bus}",
                obj_property="kvar",
                val="0.0",
            )
        )
    return CommandList(root=commands)


class EVCSFederate:

    def __init__(
        self,
        federate_name,
        input_mapping,
        broker_config: BrokerConfig,
        evcs_bus: list = None,
        ev_params: dict = None,
        control_mode: str = "dopf",
        test_mode: bool = False,
    ):
        """Initialize federate with name and remaps input into subscriptions.

        test_mode : bool
            If True, starts a local HELICS broker for standalone testing
        """
        self.evcs_bus = evcs_bus if evcs_bus is not None else ["48.1"]
        self.control_mode = control_mode
        self.broker = None
        logger.info(f"EVCS bus location(s): {self.evcs_bus}")
        logger.info(f"Control mode: {self.control_mode}")

        self.ev_params = ev_params
        if ev_params is not None:
            logger.info(
                f"Loaded EV config: {ev_params['num_evs']} EVs, "
                f"{ev_params['num_control_steps']} control steps"
            )

        if test_mode:
            self.start_broker(1)

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

        self.pub_change_commands = self.vfed.register_publication(
            "change_commands", h.HELICS_DATA_TYPE_STRING, ""
        )

        self.network = None

    def start_broker(self, n_federates=1):
        """Start a local HELICS broker for standalone test."""
        logger.info("Starting local HELICS broker (test mode)")
        initstring = f"-f {n_federates} --name=mainbroker"
        self.broker = h.helicsCreateBroker("zmq", "", initstring)
        assert h.helicsBrokerIsConnected(self.broker) == 1, "Broker connection failed"
        logger.info("Local broker created")

    def simulate(self, num_timesteps=None):
        """Lifecycle test: run finite timesteps without requiring real data.

        skips algorithm if no data is received.
        """
        if num_timesteps is None:
            num_timesteps = 4
        self.vfed.enter_executing_mode()
        logger.info("Entered execution mode (simulate)")

        granted_time = 0
        granted_time = h.helicsFederateRequestTime(self.vfed, granted_time)

        while granted_time < num_timesteps:
            if self.sub_power_P.is_updated():
                logger.info(f"Timestep {granted_time}: data received")
            else:
                logger.info(f"Timestep {granted_time}: no data (lifecycle test)")
            granted_time = h.helicsFederateRequestTime(self.vfed, granted_time + 1)

        logger.info("Completed simulation")

    def finalize(self):
        h.helicsFederateDisconnect(self.vfed)
        h.helicsFederateFree(self.vfed)
        h.helicsCloseLibrary()
        logger.info("Federate finalized")

    def _read_voltages(self):
        if not self.sub_voltages_real.is_updated():
            return None, False
        try:
            voltages_real = VoltagesReal.model_validate(self.sub_voltages_real.json)
            voltage_ids = list(voltages_real.ids)
            voltages_raw = np.array(voltages_real.values)
            if (
                hasattr(self, "_topology_base_voltages")
                and self._topology_base_voltages is not None
            ):
                base_voltages = voltages_raw / self._topology_base_voltages
            else:
                v_base = np.median(voltages_raw)
                base_voltages = (
                    voltages_raw / v_base if v_base > 0 else voltages_raw
                )
            logger.info(
                f"Received {len(base_voltages)} voltages from feeder (converted to pu)"
            )
            vr_by_id = {str(k).lower(): v for k, v in zip(voltage_ids, voltages_raw)}
            vi_by_id = {}
            try:
                voltages_imag = VoltagesImaginary.model_validate(
                    self.sub_voltages_imag.json
                )
                vi_by_id = {
                    str(k).lower(): v
                    for k, v in zip(voltages_imag.ids, voltages_imag.values)
                }
            except Exception:
                vi_by_id = {}
            base_v_by_id = getattr(self, "_base_v_by_id", {})
            measured_pu = {}
            for bus in self.evcs_bus:
                key = bus.lower()
                if key not in vr_by_id:
                    continue
                vr = float(vr_by_id[key])
                vi = float(vi_by_id.get(key, 0.0))
                base_v = float(base_v_by_id.get(key, 0.0))
                if base_v > 0:
                    measured_pu[bus] = float(np.hypot(vr, vi) / base_v)
            if measured_pu:
                self._evcs_measured_pu = measured_pu
                logger.info(f"Measured EVCS-bus voltages (pu): {measured_pu}")
            all_bus_pu = {}
            for k, vrv in vr_by_id.items():
                bv = float(base_v_by_id.get(k, 0.0))
                if bv > 0:
                    all_bus_pu[k] = float(np.hypot(vrv, float(vi_by_id.get(k, 0.0))) / bv)
            if all_bus_pu:
                self._all_bus_pu = all_bus_pu
            return base_voltages, True
        except Exception as e:
            logger.warning(f"Could not parse voltages: {e}")
            return None, False

    def _tou_price(self, hour):
        return 0.20 if 16.0 <= hour < 21.0 else 0.08

    def _accumulate_metrics(self):
        pu = getattr(self, "_all_bus_pu", None)
        if pu:
            lo = min(pu.values())
            hi = max(pu.values())
            self._m_min_pu = lo if self._m_min_pu is None else min(self._m_min_pu, lo)
            self._m_max_pu = hi if self._m_max_pu is None else max(self._m_max_pu, hi)
            self._m_n_uv += sum(1 for v in pu.values() if v < 0.95)
            self._m_n_ov += sum(1 for v in pu.values() if v > 1.05)
            self._m_vcells += len(pu)
        dl = self._last_delivered_ev
        if dl is not None:
            self._m_steps += 1
            dsum = float(sum(dl))
            prop = self._last_proposed_ev if self._last_proposed_ev is not None else dl
            psum = float(sum(prop))
            self._m_peak_ev = max(self._m_peak_ev, dsum)
            dt = self._m_dt_h
            price = getattr(self, "_last_price", self._tou_price(self._last_hour))
            self._m_energy_deliv += dsum * dt
            self._m_energy_prop += psum * dt
            self._m_cost_deliv += dsum * dt * price
            self._m_cost_prop += psum * dt * price
            if self._m_steps % 10 == 0:
                saved = self._m_cost_prop - self._m_cost_deliv
                lred = self._m_losskwh_base - self._m_losskwh_deliv
                logger.info(
                    f"[BENEFIT so far] charging cost saved ${saved:.2f} | "
                    f"est. network loss reduction {lred:.2f} kWh"
                )

    def _log_summary(self):
        mode = getattr(self, "control_mode", "?")
        deliv = self._m_energy_deliv
        prop = self._m_energy_prop
        cd = self._m_cost_deliv
        cp = self._m_cost_prop
        cost_delta = cp - cd
        cost_pct = (100.0 * cost_delta / cp) if cp > 0 else 0.0
        minpu = self._m_min_pu if self._m_min_pu is not None else float("nan")
        maxpu = self._m_max_pu if self._m_max_pu is not None else float("nan")
        pre = self._m_precurtail_min_pu
        logger.info("")
        logger.info("=" * 60)
        logger.info("===== EV SIMULATION SUMMARY =====")
        logger.info(f"control_mode        : {mode}")
        logger.info(f"timesteps           : {self._m_steps}")
        logger.info(f"min voltage (pu)    : {minpu:.4f}")
        logger.info(f"max voltage (pu)    : {maxpu:.4f}")
        vcells = self._m_vcells
        viol_rate = (100.0 * (self._m_n_uv + self._m_n_ov) / vcells) if vcells else 0.0
        logger.info(f"voltage violations  : {self._m_n_uv} undervolt, {self._m_n_ov} overvolt ({viol_rate:.2f}% of bus-steps)")
        logger.info(f"peak EV load (kW)   : {self._m_peak_ev:.1f}")
        logger.info(f"EV energy delivered : {deliv:.1f} kWh")
        logger.info(f"EV charging cost    : ${cd:.2f}")
        logger.info("----- BENEFITS vs uncontrolled baseline -----")
        logger.info(f"COST REDUCTION      : ${cost_delta:.2f} ({cost_pct:.1f}%)  [${cd:.2f} with control vs ${cp:.2f} uncontrolled, TOU tariff]")
        avg_pd = cd / deliv if deliv > 0 else 0.0
        avg_pp = cp / prop if prop > 0 else 0.0
        logger.info(f"avg price paid      : ${avg_pd:.4f}/kWh with control vs ${avg_pp:.4f}/kWh uncontrolled")
        lb = self._m_losskwh_base
        ld = self._m_losskwh_deliv
        if lb > 0.0:
            lpct = 100.0 * (lb - ld) / lb
            logger.info(f"LOSS REDUCTION      : {lb - ld:.2f} kWh ({lpct:.1f}%)  [Zbus model estimate: {ld:.2f} kWh with control vs {lb:.2f} kWh uncontrolled]")
        elif self._m_sumsq_base > 0.0:
            lpct = 100.0 * (self._m_sumsq_base - self._m_sumsq_deliv) / self._m_sumsq_base
            logger.info(f"LOSS REDUCTION      : {lpct:.1f}% est.  [quadratic demand proxy]")
        if pre is not None:
            logger.info(f"min pu w/o control  : {pre:.4f} (measured pre-curtailment, est)  vs {minpu:.4f} controlled")
        else:
            logger.info("min pu w/o control  : run control_mode=uncontrolled for the exact voltage baseline")
        logger.info("=" * 60)

    def run(self):
        """Main run loop: build LinearizedNetwork from topology, then run PSO each timestep."""
        logger.info(f"Federate connected: {datetime.now()}")
        logger.info("=" * 60)
        logger.info("EVCS FEDERATE")
        logger.info(f"  Target buses: {self.evcs_bus}")

        evcs_bus_assignment = (
            self.ev_params.get("evcs_bus_assignment", {}) if self.ev_params else {}
        )
        for bus, evs in evcs_bus_assignment.items():
            if evs:
                logger.info(
                    f"    Bus {bus}: {len(evs)} EVs (indices {evs[0]}-{evs[-1]})"
                )
        logger.info("=" * 60)

        self.vfed.enter_executing_mode()

        iterate_if_needed = h.HELICS_ITERATION_REQUEST_ITERATE_IF_NEEDED
        no_iteration = h.HELICS_ITERATION_REQUEST_NO_ITERATION
        voltage_control = self.control_mode in ("curtailment", "ymatrix")
        max_curtail_iters = 10
        no_voltage_limit = 3

        granted_time, iteration_state = h.helicsFederateRequestTimeIterative(
            self.vfed, h.HELICS_TIME_MAXTIME, iterate_if_needed
        )

        num_particles = 20
        max_iterations = 12
        timestep_count = 0
        network_built = False
        self._proposed_ev = []
        self._prev_curtailed = []
        self._curtail_iter = 0
        self._no_voltage_count = 0
        self._last_delivered_ev = None
        self._last_proposed_ev = None
        self._last_hour = 12.0
        self._m_min_pu = None
        self._m_max_pu = None
        self._m_n_uv = 0
        self._m_n_ov = 0
        self._m_peak_ev = 0.0
        self._m_energy_deliv = 0.0
        self._m_energy_prop = 0.0
        self._m_cost_deliv = 0.0
        self._m_cost_prop = 0.0
        self._m_steps = 0
        self._m_vcells = 0
        self._m_precurtail_min_pu = None
        self._m_dt_h = 0.25
        self._m_losskwh_deliv = 0.0
        self._m_losskwh_base = 0.0
        self._m_sumsq_deliv = 0.0
        self._m_sumsq_base = 0.0

        while granted_time < h.HELICS_TIME_MAXTIME:
            if iteration_state == h.HELICS_ITERATION_RESULT_ITERATING:
                if not voltage_control:
                    granted_time, iteration_state = h.helicsFederateRequestTimeIterative(
                        self.vfed, h.HELICS_TIME_MAXTIME, no_iteration
                    )
                    continue
                _, updated = self._read_voltages()
                if updated:
                    self._no_voltage_count = 0
                else:
                    self._no_voltage_count += 1
                ev_load_values, ev_load_per_bus = self._apply_curtailment(
                    self._proposed_ev
                )
                cmd_list = build_change_commands(
                    self.evcs_bus, ev_load_values
                )
                self.pub_change_commands.publish(cmd_list.model_dump_json())
                self._last_delivered_ev = list(ev_load_values)
                if self._curtail_iter == 0:
                    _pu = getattr(self, "_all_bus_pu", None)
                    if _pu:
                        _lo = min(_pu.values())
                        self._m_precurtail_min_pu = _lo if self._m_precurtail_min_pu is None else min(self._m_precurtail_min_pu, _lo)
                self._curtail_iter += 1
                max_delta = max(
                    (abs(a - b) for a, b in zip(ev_load_values, self._prev_curtailed)),
                    default=0.0,
                )
                self._prev_curtailed = ev_load_values
                measured = getattr(self, "_evcs_measured_pu", {})
                all_healthy = bool(measured) and all(
                    v >= 0.95 for v in measured.values()
                )
                logger.info(
                    f"[ITER] round {self._curtail_iter}: max dP={max_delta:.3f} kW, "
                    f"healthy={all_healthy}, total={sum(ev_load_values):.2f} kW"
                )
                done = (
                    all_healthy
                    or self._curtail_iter >= max_curtail_iters
                    or self._no_voltage_count >= no_voltage_limit
                )
                if done:
                    granted_time, iteration_state = h.helicsFederateRequestTimeIterative(
                        self.vfed, h.HELICS_TIME_MAXTIME, no_iteration
                    )
                else:
                    granted_time, iteration_state = h.helicsFederateRequestTimeIterative(
                        self.vfed, int(granted_time) + 1, iterate_if_needed
                    )
                continue

            if not self.sub_power_P.is_updated():
                granted_time, iteration_state = h.helicsFederateRequestTimeIterative(
                    self.vfed, h.HELICS_TIME_MAXTIME, iterate_if_needed
                )
                continue

            timestep_count += 1
            logger.info("")
            logger.info("=" * 60)
            logger.info(f"TIMESTEP {timestep_count} | HELICS Time: {granted_time}")
            logger.info("=" * 60)

            if not network_built and self.sub_topology.is_updated():
                try:
                    topology = Topology.model_validate(self.sub_topology.json)
                    logger.info("Received topology from feeder")

                    base_volt_raw = np.array(topology.base_voltage_magnitudes.values)
                    self._topology_base_voltages = base_volt_raw
                    base_volt_pu = np.ones_like(base_volt_raw)

                    self.network = LinearizedNetwork(
                        bus_ids=list(topology.base_voltage_magnitudes.ids),
                        base_voltages=base_volt_pu,
                        slack_bus=topology.slack_bus[0] if topology.slack_bus else None,
                    )
                    self.network.build_from_topology(topology)
                    network_built = True
                    logger.info(
                        f"Built LinearizedNetwork with {self.network.n_buses} buses"
                    )

                    self._base_v_by_id = {
                        str(bid).lower(): bval
                        for bid, bval in zip(
                            topology.base_voltage_magnitudes.ids,
                            topology.base_voltage_magnitudes.values,
                        )
                    }

                    if self.control_mode in ("dopf", "ymatrix"):
                        try:
                            self.network.build_sensitivity_from_admittance(
                                topology,
                                diagonal_only=(self.control_mode == "dopf"),
                                sensitivity_scale=float(
                                    (self.ev_params or {}).get(
                                        "voltage_sensitivity_scale", 1.0
                                    )
                                ),
                            )
                            logger.info(
                                f"Control mode {self.control_mode}: Zbus dV/dP from "
                                f"admittance (diagonal_only={self.control_mode == 'dopf'})"
                            )
                        except Exception as ye:
                            logger.warning(
                                f"admittance sensitivity failed, keeping "
                                f"LinDistFlow sensitivity: {ye}"
                            )
                except Exception as e:
                    logger.warning(f"Could not build network from topology: {e}")
                    self.network = None

            base_voltages, _ = self._read_voltages()
            self._last_hour = (float(int(granted_time)) * self._m_dt_h) % 24.0
            self._accumulate_metrics()

            if base_voltages is None:
                if self.network is not None:
                    base_voltages = np.ones(self.network.n_buses)
                else:
                    base_voltages = np.ones(100)

            power_P = PowersReal.model_validate(self.sub_power_P.json)
            power_Q = PowersImaginary.model_validate(self.sub_power_Q.json)

            load_ids = list(power_P.ids)
            logger.info(f"[INPUT] Received feeder data: {len(load_ids)} load buses")
            self._track_losses(power_P)

            time = power_P.time
            time_idx = int(granted_time)

            if not hasattr(self, "_uncontrolled_rate"):
                ep = self.ev_params
                _, self._uncontrolled_rate = ev_simulation.uncontrolled_charging(
                    ep["initial_soc"],
                    ep["num_control_steps"],
                    ep["control_interval"],
                    ep["battery_capacity"],
                    ep["charging_efficiency"],
                    ep["arrival_time_idx"],
                    ep["departure_time_idx"],
                    ep["num_evs"],
                    ep["max_charging_rate"],
                    ep["desired_soc"],
                )

            pso_start = time_module.time()

            if self.control_mode in ("uncontrolled", "curtailment"):
                charging_rate = self._uncontrolled_rate
                true_cost = 0.0
                logger.info(f"[UNCONTROLLED] Using greedy charging (no PSO)")
            else:
                logger.info(
                    f"[PSO] Starting optimization: {num_particles} particles, {max_iterations} iterations"
                )
                charging_rate, true_cost = ev_simulation.ev_pso_optimization(
                    num_particles,
                    max_iterations,
                    self.network,
                    base_voltages,
                    self.evcs_bus,
                    ev_params=self.ev_params,
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
                logger.info(
                    f"[RESULT] Bus {bus}: {num_charging} EVs charging, Power: {bus_power:.2f} kW"
                )

            total_ev_load = sum(ev_load_values)
            logger.info(f"[RESULT] Total across all buses: {total_ev_load:.2f} kW")

            proposed_ev = list(ev_load_values)

            baseline_ev = []
            for bus in self.evcs_bus:
                ev_indices = evcs_bus_assignment.get(bus, [])
                if ev_indices:
                    baseline_ev.append(
                        float(np.sum(self._uncontrolled_rate[ev_indices, time_idx]))
                    )
                else:
                    baseline_ev.append(0.0)

            cmd_list = build_change_commands(
                self.evcs_bus, ev_load_values
            )
            self.pub_change_commands.publish(cmd_list.model_dump_json())
            self._last_delivered_ev = list(ev_load_values)
            self._last_proposed_ev = baseline_ev
            self._last_price = float(self.ev_params["electricity_price"][time_idx])
            logger.info(f"[OUTPUT] Published change_commands: {ev_load_per_bus}")
            logger.info("-" * 60)

            self._proposed_ev = proposed_ev
            self._prev_curtailed = list(ev_load_values)
            self._curtail_iter = 0
            self._no_voltage_count = 0

            if voltage_control:
                granted_time, iteration_state = h.helicsFederateRequestTimeIterative(
                    self.vfed, int(granted_time) + 1, iterate_if_needed
                )
            else:
                granted_time, iteration_state = h.helicsFederateRequestTimeIterative(
                    self.vfed, h.HELICS_TIME_MAXTIME, no_iteration
                )

        self._accumulate_metrics()
        self._log_summary()
        self.stop()

    def _track_losses(self, power_P):
        d_prev = self._last_delivered_ev
        b_prev = self._last_proposed_ev
        if d_prev is None or b_prev is None:
            return
        vals = np.array(power_P.values, dtype=float)
        dt = self._m_dt_h
        demand = float(-np.sum(vals[vals < 0.0]))
        d_sum = float(sum(d_prev))
        b_sum = float(sum(b_prev))
        demand_base = max(demand - d_sum + b_sum, 0.0)
        self._m_sumsq_deliv += demand * demand * dt
        self._m_sumsq_base += demand_base * demand_base * dt
        net = self.network
        if net is None or getattr(net, "dV_dP", None) is None:
            return
        if self.control_mode not in ("dopf", "ymatrix"):
            return
        ids = list(power_P.ids)
        idx = getattr(self, "_loss_idx", None)
        if idx is None or len(idx) != len(ids):
            pos = {str(b).lower(): i for i, b in enumerate(net.bus_ids)}
            idx = [pos.get(str(k).lower()) for k in ids]
            self._loss_idx = idx
            self._loss_ev_idx = [pos.get(str(b).lower()) for b in self.evcs_bus]
        q = np.zeros(net.n_buses)
        for i, v in zip(idx, vals):
            if i is not None:
                q[i] += float(v) / 1000.0
        R = -np.asarray(net.dV_dP)
        qb = q.copy()
        for i, dv, bv in zip(self._loss_ev_idx, d_prev, b_prev):
            if i is not None:
                qb[i] += (float(dv) - float(bv)) / 1000.0
        loss_d = float(q @ R @ q) * 1000.0
        loss_b = float(qb @ R @ qb) * 1000.0
        if np.isfinite(loss_d) and np.isfinite(loss_b):
            self._m_losskwh_deliv += max(loss_d, 0.0) * dt
            self._m_losskwh_base += max(loss_b, 0.0) * dt

    def _apply_curtailment(self, proposed):
        measured = getattr(self, "_evcs_measured_pu", None)
        prev = getattr(self, "_prev_curtailed", None)
        if not measured or not prev or len(prev) != len(proposed):
            return list(proposed), dict(zip(self.evcs_bus, proposed))
        new_values = []
        for bus, p_prop, p_prev in zip(self.evcs_bus, proposed, prev):
            v_pu = measured.get(bus)
            if v_pu is not None and v_pu < 0.95:
                p_new = max(0.0, p_prev - 0.25 * p_prop)
                logger.info(
                    f"[CURTAIL] Bus {bus}: v={v_pu:.4f}pu < 0.95, "
                    f"{p_prev:.2f} kW -> {p_new:.2f} kW"
                )
            else:
                p_new = p_prev
            new_values.append(p_new)
        return new_values, dict(zip(self.evcs_bus, new_values))

    def stop(self):
        h.helicsFederateDisconnect(self.vfed)
        h.helicsFederateFree(self.vfed)
        h.helicsCloseLibrary()
        logger.info(f"Federate disconnected: {datetime.now()}")


def run_simulator(broker_config: BrokerConfig):
    logger.info("Starting run_simulator")
    with open("static_inputs.json") as f:
        config = json.load(f)
        federate_name = config["name"]
        evcs_bus = config.get("evcs_bus", ["48.1"])
        control_mode = config.get("control_mode", "dopf")
        logger.info(f"Loaded evcs_bus from config: {evcs_bus}")
        logger.info(f"Control mode from config: {control_mode}")

    ev_params = generate_ev_parameters(config)
    ev_params["voltage_sensitivity_scale"] = config.get("voltage_sensitivity_scale", 1.0)
    logger.info(
        f"Generated EV parameters: {ev_params['num_evs']} EVs across "
        f"{len(ev_params['num_evs_per_station'])} stations"
    )

    with open("input_mapping.json") as f:
        input_mapping = json.load(f)

    try:
        sfed = EVCSFederate(
            federate_name,
            input_mapping,
            broker_config,
            evcs_bus,
            ev_params=ev_params,
            control_mode=control_mode,
        )
        logger.info("Value federate created")
    except h.HelicsException as e:
        logger.error(f"Failed to create HELICS Value Federate: {str(e)}")
        return

    sfed.run()
    logger.info("run_simulator complete")


if __name__ == "__main__":
    run_simulator(BrokerConfig(broker_ip="127.0.0.1"))
