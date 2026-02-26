"""Core class to abstract OpenDSS into Feeder class."""

import json
import logging
import math
import os
import random
import time
from enum import Enum
from time import strptime
from typing import Dict, List, Optional, Set

import boto3
import numpy as np
import opendssdirect as dss
import xarray as xr
from botocore import UNSIGNED
from botocore.config import Config
from .dss_functions import (
    get_capacitors,
    get_generators,
    get_loads,
    get_pvsystems,
    get_voltages,
)

from oedisi.types.data_types import (
    Command,
    InverterControl,
    InverterControlMode,
    IncidenceList,
)
from pydantic import BaseModel
from scipy.sparse import coo_matrix, csc_matrix

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


def permutation(from_list, to_list):
    """Return permutation such that to_list[permute] == from_list.

    permute[i] = j means from_list[i] = to_list[j].
    """
    index_map = {v: i for i, v in enumerate(to_list)}
    return [index_map[v] for v in from_list]


class FeederConfig(BaseModel):
    """JSON configuration for feeder simulation."""

    name: str
    use_smartds: bool = False
    user_uploads_model: bool = False
    profile_location: str
    opendss_location: str
    existing_feeder_file: Optional[str] = None
    sensor_location: Optional[str] = None
    start_date: str
    number_of_timesteps: int
    run_freq_sec: float = 15 * 60
    start_time_index: int = 0
    topology_output: str = "topology.json"
    use_sparse_admittance: bool = False
    tap_setting: Optional[int] = None
    pv_scale_factor: float = 1.0


class FeederMapping(BaseModel):
    static_inputs: FeederConfig
    input_mapping: Dict[str, str]


class OpenDSSState(Enum):
    """Enum of all OpenDSSStates traversed in a simulation."""

    UNLOADED = 1
    LOADED = 2
    SNAPSHOT_RUN = 3
    SOLVE_AT_TIME = 4
    DISABLED_RUN = 5
    DISABLED_SOLVE = 6
    DISABLED = 7


class FeederSimulator(object):
    """Wraps OpenDSS circuit for OEDISI co-simulation."""

    _feeder_file: str
    _AllNodeNames: List[str]
    _source_indexes: List[int]
    _nodes_index: List[int]
    _name_index_dict: Dict[str, int]
    _inverter_to_pvsystems: Dict[str, Set[str]]
    _pvsystem_to_inverter: Dict[str, str]
    _pvsystems: Set[str]
    _inverters: Set[str]
    _inverter_counter: int
    _xycurve_counter: int

    def __init__(self, config: FeederConfig):
        """Create a FeederSimulator object."""
        self._state = OpenDSSState.UNLOADED
        self._opendss_location = config.opendss_location
        self._profile_location = config.profile_location
        self._sensor_location = config.sensor_location
        self._use_smartds = config.use_smartds
        self._user_uploads_model = config.user_uploads_model
        self._inverter_to_pvsystems = {}
        self._pvsystem_to_inverter = {}
        self._inverters = set()
        self._inverter_counter = 0
        self._xycurve_counter = 0

        self._start_time = int(
            time.mktime(strptime(config.start_date, "%Y-%m-%d %H:%M:%S"))
        )
        self._run_freq_sec = config.run_freq_sec
        self._simulation_step = config.start_time_index
        self._number_of_timesteps = config.number_of_timesteps
        self._vmult = 0.001

        self.tap_setting = config.tap_setting
        self._pv_scale_factor = config.pv_scale_factor

        self._simulation_time_step = "15m"
        if config.existing_feeder_file is None:
            if self._use_smartds:
                self._feeder_file = os.path.join("opendss", "Master.dss")
                self.download_data("oedi-data-lake", update_loadshape_location=True)
            elif not self._use_smartds and not self._user_uploads_model:
                self._feeder_file = os.path.join("opendss", "master.dss")
                self.download_data("gadal")
            else:
                raise Exception("Set existing_feeder_file when uploading data")
        else:
            self._feeder_file = config.existing_feeder_file

        self.load_feeder()

        if self._pv_scale_factor != 1.0:
            self._scale_pv_systems(self._pv_scale_factor)

        if self._sensor_location is None:
            self.create_measurement_lists()

        self.snapshot_run()
        assert self._state == OpenDSSState.SNAPSHOT_RUN, f"{self._state}"

    def snapshot_run(self):
        """Run snapshot solve without specifying time, used for initialization."""
        assert self._state != OpenDSSState.UNLOADED, f"{self._state}"
        self.reenable()
        dss.Text.Command("CalcVoltageBases")
        dss.Text.Command("solve mode=snapshot")
        self._state = OpenDSSState.SNAPSHOT_RUN

    def reenable(self):
        dss.Text.Command("Batchedit Load..* enabled=yes")
        dss.Text.Command("Batchedit Vsource..* enabled=yes")
        dss.Text.Command("Batchedit Isource..* enabled=yes")
        dss.Text.Command("Batchedit Generator..* enabled=yes")
        dss.Text.Command("Batchedit PVsystem..* enabled=yes")
        dss.Text.Command("Batchedit Capacitor..* enabled=yes")
        dss.Text.Command("Batchedit Storage..* enabled=no")

    def download_data(self, bucket_name, update_loadshape_location=False):
        """Download data from S3 bucket."""
        logging.info(f"Downloading from bucket {bucket_name}")
        # Equivalent to --no-sign-request
        s3_resource = boto3.resource("s3", config=Config(signature_version=UNSIGNED))
        bucket = s3_resource.Bucket(bucket_name)
        opendss_location = self._opendss_location
        profile_location = self._profile_location
        sensor_location = self._sensor_location

        for obj in bucket.objects.filter(Prefix=opendss_location):
            output_location = os.path.join(
                "opendss", obj.key.replace(opendss_location, "").strip("/")
            )
            os.makedirs(os.path.dirname(output_location), exist_ok=True)
            bucket.download_file(obj.key, output_location)

        modified_loadshapes = ""
        os.makedirs(os.path.join("profiles"), exist_ok=True)
        if update_loadshape_location:
            all_profiles = set()
            with open(os.path.join("opendss", "LoadShapes.dss"), "r") as fp_loadshapes:
                for row in fp_loadshapes.readlines():
                    new_row = row.replace("../", "")
                    new_row = new_row.replace("file=", "file=../")
                    for token in new_row.split(" "):
                        if token.startswith("(file="):
                            location = (
                                token.split("=../profiles/")[1].strip().strip(")")
                            )
                            all_profiles.add(location)
                    modified_loadshapes = modified_loadshapes + new_row
            with open(os.path.join("opendss", "LoadShapes.dss"), "w") as fp_loadshapes:
                fp_loadshapes.write(modified_loadshapes)
            for profile in all_profiles:
                s3_location = f"{profile_location}/{profile}"
                bucket.download_file(s3_location, os.path.join("profiles", profile))
        else:
            for obj in bucket.objects.filter(Prefix=profile_location):
                output_location = os.path.join(
                    "profiles", obj.key.replace(profile_location, "").strip("/")
                )
                os.makedirs(os.path.dirname(output_location), exist_ok=True)
                bucket.download_file(obj.key, output_location)

        if sensor_location is not None:
            output_location = os.path.join("sensors", os.path.basename(sensor_location))
            if not os.path.exists(os.path.dirname(output_location)):
                os.makedirs(os.path.dirname(output_location))
            bucket.download_file(sensor_location, output_location)

    def create_measurement_lists(
        self,
        percent_voltage=75,
        percent_real=75,
        voltage_seed=1,
        real_seed=2,
        reactive_seed=3,
    ):
        """Initialize list of sensor locations for the measurement federate."""
        random.seed(voltage_seed)
        os.makedirs("sensors", exist_ok=True)
        voltage_subset = random.sample(
            self._AllNodeNames,
            math.floor(len(self._AllNodeNames) * float(percent_voltage) / 100),
        )
        with open(os.path.join("sensors", "voltage_ids.json"), "w") as fp:
            json.dump(voltage_subset, fp, indent=4)

        random.seed(real_seed)
        real_subset = random.sample(
            self._AllNodeNames,
            math.floor(len(self._AllNodeNames) * float(percent_real) / 100),
        )
        with open(os.path.join("sensors", "real_ids.json"), "w") as fp:
            json.dump(real_subset, fp, indent=4)

        random.seed(reactive_seed)
        reactive_subset = random.sample(
            self._AllNodeNames,
            math.floor(len(self._AllNodeNames) * float(percent_voltage) / 100),
        )
        with open(os.path.join("sensors", "reactive_ids.json"), "w") as fp:
            json.dump(reactive_subset, fp, indent=4)

    def get_circuit_name(self):
        """Get name of current opendss circuit."""
        return self._circuit.Name()

    def get_source_indices(self):
        """Get indices of slack buses."""
        return self._source_indexes

    def get_node_names(self):
        """Get node names in order."""
        return self._AllNodeNames

    def load_feeder(self):
        """Load feeder once downloaded."""
        try:
            dss.Basic.LegacyModels(True)
        except Exception:
            pass  # LegacyModels not supported in newer OpenDSS versions
        dss.Text.Command("clear")
        dss.Text.Command("redirect " + self._feeder_file)
        result = dss.Text.Result()
        if not result == "":
            raise ValueError("Feeder not loaded: " + result)
        self._circuit = dss.Circuit
        self._AllNodeNames = self._circuit.YNodeOrder()
        self._node_number = len(self._AllNodeNames)
        self._nodes_index = [self._AllNodeNames.index(ii) for ii in self._AllNodeNames]
        self._name_index_dict = {
            ii: self._AllNodeNames.index(ii) for ii in self._AllNodeNames
        }

        self._source_indexes = []
        for Source in dss.Vsources.AllNames():
            self._circuit.SetActiveElement("Vsource." + Source)
            Bus = dss.CktElement.BusNames()[0].upper()
            for phase in range(1, dss.CktElement.NumPhases() + 1):
                self._source_indexes.append(
                    self._AllNodeNames.index(Bus.upper() + "." + str(phase))
                )

        self.setup_vbase()

        self._pvsystems = set()
        for PV in get_pvsystems(dss):
            self._pvsystems.add("PVSystem." + PV["name"])

        if self.tap_setting is not None:
            # Does not work with AutoTrans or 3-winding transformers.
            dss.Text.Command(f"batchedit transformer..* wdg=2 tap={self.tap_setting}")
        self._state = OpenDSSState.LOADED

    def _scale_pv_systems(self, scale_factor):
        """Scale all PV systems by the given factor after model loading."""
        logging.info(f"Scaling PV systems by factor {scale_factor}")
        pv_flag = dss.PVsystems.First()
        while pv_flag:
            name = dss.PVsystems.Name()
            old_pmpp = dss.PVsystems.Pmpp()
            old_kva = dss.PVsystems.kVARated()
            new_pmpp = old_pmpp * scale_factor
            new_kva = old_kva * scale_factor
            dss.Text.Command(f"Edit PVSystem.{name} Pmpp={new_pmpp:.1f} kVA={new_kva:.1f}")
            logging.info(f"  PVSystem.{name}: Pmpp {old_pmpp:.0f} -> {new_pmpp:.0f} kW, kVA {old_kva:.0f} -> {new_kva:.0f}")
            pv_flag = dss.PVsystems.Next()

    def disable_elements(self):
        """Disable most elements, used in disabled_run."""
        assert self._state != OpenDSSState.UNLOADED, f"{self._state}"
        dss.Text.Command("batchedit regcontrol..* enabled=false")
        dss.Text.Command("batchedit vsource..* enabled=false")
        dss.Text.Command("batchedit isource..* enabled=false")
        dss.Text.Command("batchedit load..* enabled=false")
        dss.Text.Command("batchedit generator..* enabled=false")
        dss.Text.Command("batchedit pvsystem..* enabled=false")
        dss.Text.Command("Batchedit Capacitor..* enabled=false")
        dss.Text.Command("batchedit storage..* enabled=false")
        self._state = OpenDSSState.DISABLED

    def disabled_run(self):
        """Disable most elements and solve, used for Y-matrix computation."""
        self.disable_elements()
        assert self._state == OpenDSSState.DISABLED, f"{self._state}"
        dss.Text.Command("CalcVoltageBases")
        dss.Text.Command("set maxiterations=20")
        dss.Text.Command("solve")
        self._state = OpenDSSState.DISABLED_RUN

    def get_y_matrix(self):
        """Calculate Y-matrix as a coo-matrix."""
        self.disabled_run()
        self._state = OpenDSSState.DISABLED_RUN

        Ysparse = csc_matrix(dss.YMatrix.getYsparse())
        Ymatrix = Ysparse.tocoo()
        new_order = self._circuit.YNodeOrder()
        permute = np.array(permutation(new_order, self._AllNodeNames))
        return coo_matrix(
            (Ymatrix.data, (permute[Ymatrix.row], permute[Ymatrix.col])),
            shape=Ymatrix.shape,
        )

    def get_load_y_matrix(self):
        """Calculate Y-matrix with only loads enabled."""
        assert self._state == OpenDSSState.SOLVE_AT_TIME, f"{self._state}"
        self.disable_elements()
        dss.Text.Command("batchedit Load..* enabled=true")
        dss.Text.Command("CalcVoltageBases")
        dss.Text.Command("set maxiterations=20")
        dss.Text.Command("solve")

        Ysparse = csc_matrix(dss.YMatrix.getYsparse())
        Ymatrix = Ysparse.tocoo()
        new_order = self._circuit.YNodeOrder()
        permute = np.array(permutation(new_order, self._AllNodeNames))

        dss.Text.Command("batchedit Load..* enabled=false")
        self._state = OpenDSSState.DISABLED_RUN
        self.reenable()

        dss.Text.Command("CalcVoltageBases")
        dss.Text.Command("set maxiterations=20")
        dss.Text.Command("solve")
        self._state = OpenDSSState.SOLVE_AT_TIME

        return coo_matrix(
            (Ymatrix.data, (permute[Ymatrix.row], permute[Ymatrix.col])),
            shape=Ymatrix.shape,
        )

    def setup_vbase(self):
        """Load base voltages into feeder."""
        self._Vbase_allnode = np.zeros((self._node_number), dtype=np.complex128)
        self._Vbase_allnode_dict = {}
        for ii, node in enumerate(self._AllNodeNames):
            self._circuit.SetActiveBus(node)
            self._Vbase_allnode[ii] = dss.Bus.kVBase() * 1000
            self._Vbase_allnode_dict[node] = self._Vbase_allnode[ii]

    def initial_disabled_solve(self):
        """Solve at time 0 when run is disabled."""
        assert self._state == OpenDSSState.DISABLED_RUN, f"{self._state}"
        hour = 0
        second = 0
        dss.Text.Command(
            f"set mode=yearly loadmult=1 number=1 hour={hour} sec={second} "
            f"stepsize=0"
        )
        dss.Text.Command("solve")
        self._state = OpenDSSState.DISABLED_SOLVE

    def just_solve(self):
        """Solve without setting time, useful after change_obj commands."""
        assert (
            self._state != OpenDSSState.UNLOADED
            and self._state != OpenDSSState.DISABLED_RUN
        ), f"{self._state}"
        dss.Text.Command("solve")

    def solve(self, hour, second):
        """Solve at specified time."""
        assert (
            self._state != OpenDSSState.UNLOADED
            and self._state != OpenDSSState.DISABLED_RUN
        ), f"{self._state}"

        dss.Text.Command(
            f"set mode=yearly loadmult=1 number=1 hour={hour} sec={second} "
            f"stepsize=0"
        )
        dss.Text.Command("solve")
        self._state = OpenDSSState.SOLVE_AT_TIME

    def _ready_to_load_power(self, static):
        """Assert OpenDSS state allows power retrieval."""
        if static:
            assert self._state != OpenDSSState.UNLOADED, f"{self._state}"
        else:
            assert self._state == OpenDSSState.SOLVE_AT_TIME, f"{self._state}"

    def get_PQs_load(self, static=False):
        """Get active and reactive power of loads as xarray."""
        self._ready_to_load_power(static)

        all_node_names = set(self._AllNodeNames)
        PQs: List[complex] = []
        node_names: List[str] = []
        pq_names: List[str] = []
        for ld in get_loads(dss, self._circuit):
            self._circuit.SetActiveElement("Load." + ld["name"])
            current_pq_name = dss.CktElement.Name()
            for ii in range(len(ld["phases"])):
                node_name = ld["bus1"].upper() + "." + ld["phases"][ii]
                assert (
                    node_name in all_node_names
                ), f"{node_name} for {current_pq_name} not found"
                if static:
                    power = complex(ld["kW"], ld["kVar"])
                    PQs.append(power / len(ld["phases"]))
                else:
                    power = dss.CktElement.Powers()
                    PQs.append(complex(power[2 * ii], power[2 * ii + 1]))
                pq_names.append(current_pq_name)
                node_names.append(node_name)
        pq_xr = xr.DataArray(
            PQs,
            dims=("eqnode",),
            coords={
                "equipment_ids": ("eqnode", pq_names),
                "ids": ("eqnode", node_names),
            },
        )
        return pq_xr.sortby(pq_xr.ids)

    def get_PQs_pv(self, static=False):
        """Get active and reactive power of PVSystems as xarray."""
        self._ready_to_load_power(static)

        all_node_names = set(self._AllNodeNames)
        PQs: List[complex] = []
        node_names: List[str] = []
        pq_names: List[str] = []
        for PV in get_pvsystems(dss):
            bus = PV["bus"].split(".")
            if len(bus) == 1:
                bus = bus + ["1", "2", "3"]
            self._circuit.SetActiveElement("PVSystem." + PV["name"])
            current_pq_name = dss.CktElement.Name()
            for ii in range(len(bus) - 1):
                node_name = bus[0].upper() + "." + bus[ii + 1]
                assert (
                    node_name in all_node_names
                ), f"{node_name} for {current_pq_name} not found"
                if static:
                    power = complex(
                        -1 * PV["kW"], -1 * PV["kVar"]
                    )  # -1 because injecting
                    PQs.append(power / (len(bus) - 1))
                else:
                    power = dss.CktElement.Powers()
                    PQs.append(complex(power[2 * ii], power[2 * ii + 1]))
                pq_names.append(current_pq_name)
                node_names.append(node_name)
        pq_xr = xr.DataArray(
            PQs,
            dims=("eqnode",),
            coords={
                "equipment_ids": ("eqnode", pq_names),
                "ids": ("eqnode", node_names),
            },
        )
        return pq_xr.sortby(pq_xr.ids)

    def get_PQs_gen(self, static=False):
        """Get active and reactive power of Generators as xarray."""
        self._ready_to_load_power(static)

        all_node_names = set(self._AllNodeNames)
        PQs: List[complex] = []
        node_names: List[str] = []
        pq_names: List[str] = []
        for gen in get_generators(dss):
            bus = gen["bus"].split(".")
            if len(bus) == 1:
                bus = bus + ["1", "2", "3"]
            self._circuit.SetActiveElement("Generator." + gen["name"])
            current_pq_name = dss.CktElement.Name()
            for ii in range(len(bus) - 1):
                node_name = bus[0].upper() + "." + bus[ii + 1]
                assert (
                    node_name in all_node_names
                ), f"{node_name} for {current_pq_name} not found"
                if static:
                    power = complex(
                        -1 * gen["kW"], -1 * gen["kVar"]
                    )  # -1 because injecting
                    PQs.append(power / (len(bus) - 1))
                else:
                    power = dss.CktElement.Powers()
                    PQs.append(complex(power[2 * ii], power[2 * ii + 1]))
                pq_names.append(current_pq_name)
                node_names.append(node_name)
        pq_xr = xr.DataArray(
            PQs,
            dims=("eqnode",),
            coords={
                "equipment_ids": ("eqnode", pq_names),
                "ids": ("eqnode", node_names),
            },
        )
        return pq_xr.sortby(pq_xr.ids)

    def get_PQs_cap(self, static=False):
        """Get active and reactive power of Capacitors as xarray."""
        self._ready_to_load_power(static)

        all_node_names = set(self._AllNodeNames)
        PQs: List[complex] = []
        node_names: List[str] = []
        pq_names: List[str] = []
        for cap in get_capacitors(dss):
            current_pq_name = cap["name"]
            for ii in range(cap["numPhases"]):
                node_name = cap["busname"].upper() + "." + cap["busphase"][ii]
                assert (
                    node_name in all_node_names
                ), f"{node_name} for {current_pq_name} not found"
                if static:
                    power = complex(
                        0, -1 * cap["kVar"]
                    )  # -1 because injected into the grid
                    PQs.append(power / cap["numPhases"])
                else:
                    PQs.append(complex(0, cap["power"][2 * ii + 1]))
                pq_names.append(current_pq_name)
                node_names.append(node_name)
        pq_xr = xr.DataArray(
            PQs,
            dims=("eqnode",),
            coords={
                "equipment_ids": ("eqnode", pq_names),
                "ids": ("eqnode", node_names),
            },
        )
        return pq_xr.sortby(pq_xr.ids)

    def get_base_voltages(self):
        """Get base voltages xarray."""
        return xr.DataArray(self._Vbase_allnode, {"ids": self._AllNodeNames})

    def get_disabled_solve_voltages(self):
        """Get voltage xarray when elements are disabled."""
        assert self._state == OpenDSSState.DISABLED_SOLVE, f"{self._state}"
        return self._get_voltages()

    def get_voltages_snapshot(self):
        """Get voltage xarray in snapshot run."""
        assert self._state == OpenDSSState.SNAPSHOT_RUN, f"{self._state}"
        return self._get_voltages()

    def get_voltages_actual(self):
        """Get voltages xarray at current time."""
        assert self._state == OpenDSSState.SOLVE_AT_TIME, f"{self._state}"
        return self._get_voltages()

    def _get_voltages(self):
        """Get voltages ordered by self._AllNodeNames."""
        assert (
            self._state != OpenDSSState.DISABLED_RUN
            and self._state != OpenDSSState.UNLOADED
        ), f"{self._state}"
        name_voltage_dict = get_voltages(self._circuit)
        res_feeder_voltages = np.zeros((len(self._AllNodeNames)), dtype=np.complex128)
        for voltage_name in name_voltage_dict.keys():
            res_feeder_voltages[self._name_index_dict[voltage_name]] = (
                name_voltage_dict[voltage_name]
            )
        return xr.DataArray(
            res_feeder_voltages, {"ids": list(name_voltage_dict.keys())}
        )

    def apply_power_injection(self, power_real, power_imag):
        """Apply power injections to the feeder, skipping missing buses with a warning."""
        assert self._state != OpenDSSState.UNLOADED, f"{self._state}"

        for bus_id, p_kw in zip(power_real.ids, power_real.values):
            try:
                dss.Circuit.SetActiveBus(bus_id)
                dss.Loads.Name(bus_id)
                dss.Loads.kW(dss.Loads.kW() + p_kw)
                logger.info(f"Applied {p_kw:.2f} kW to load at bus {bus_id}")
            except Exception as e:
                logger.warning(f"Could not apply power injection to bus {bus_id}: {e}")

        for bus_id, q_kvar in zip(power_imag.ids, power_imag.values):
            try:
                dss.Circuit.SetActiveBus(bus_id)
                dss.Loads.Name(bus_id)
                dss.Loads.kvar(dss.Loads.kvar() + q_kvar)
            except Exception as e:
                logger.warning(f"Could not apply reactive power injection to bus {bus_id}: {e}")

    def remove_power_injection(self, power_real, power_imag):
        """Remove previously applied power injections to prevent accumulation across timesteps."""
        assert self._state != OpenDSSState.UNLOADED, f"{self._state}"

        for bus_id, p_kw in zip(power_real.ids, power_real.values):
            try:
                dss.Circuit.SetActiveBus(bus_id)
                dss.Loads.Name(bus_id)
                dss.Loads.kW(dss.Loads.kW() - p_kw)
                logger.info(f"Removed {p_kw:.2f} kW from load at bus {bus_id}")
            except Exception as e:
                logger.warning(f"Could not remove power injection from bus {bus_id}: {e}")

        for bus_id, q_kvar in zip(power_imag.ids, power_imag.values):
            try:
                dss.Circuit.SetActiveBus(bus_id)
                dss.Loads.Name(bus_id)
                dss.Loads.kvar(dss.Loads.kvar() - q_kvar)
            except Exception as e:
                logger.warning(f"Could not remove reactive power injection from bus {bus_id}: {e}")

    def change_obj(self, change_commands: List[Command]):
        """Set an OpenDSS object property via command list."""
        assert self._state != OpenDSSState.UNLOADED, f"{self._state}"
        for entry in change_commands:
            dss.Circuit.SetActiveElement(entry.obj_name)
            properties = dss.CktElement.AllPropertyNames()
            element_name = dss.CktElement.Name()
            assert entry.obj_property.lower() in map(
                lambda x: x.lower(), properties
            ), f"{entry.obj_property} not in {properties} for {element_name}"
            dss.Text.Command(f"{entry.obj_name}.{entry.obj_property}={entry.val}")

    def create_inverter(self, pvsystem_set: Set[str]):
        """Create new InvControl from set of PVSystems.

        Only 1 or all PVSystems can be added; this cannot be changed after creation.
        """
        assert all(
            pvsystem not in self._pvsystem_to_inverter and pvsystem in self._pvsystems
            for pvsystem in pvsystem_set
        ), f"PVsystem(s) {pvsystem_set} is already assigned inverter or may not exist"
        name = f"InvControl.invgenerated{self._inverter_counter}"
        assert all(
            pv_name.split(".")[0].lower() == "pvsystem" for pv_name in pvsystem_set
        )
        if pvsystem_set == self._pvsystems:
            pvlist = ""
        else:
            if len(pvsystem_set) != 1:
                logging.error(
                    """Controlling multiple pvsystems manually results in unstable
                    behavior when the number of phases differ"""
                )
            pvlist = ", ".join(pv_name.split(".")[1] for pv_name in pvsystem_set)
        dss.Text.Command(f"New {name} PVsystemList=[{pvlist}]")
        self._inverter_counter += 1
        self._inverters.add(name)
        for pvsystem in pvsystem_set:
            self._pvsystem_to_inverter[pvsystem] = name
        self._inverter_to_pvsystems[name] = pvsystem_set
        return name

    def create_xy_curve(self, x, y):
        """Create XYcurve object from two equal-length lists."""
        name = f"XYcurve.xygenerated{self._xycurve_counter}"
        npts = len(x)
        assert len(x) == len(y), "Length of curves do not match"
        x_str = ",".join(str(i) for i in x)
        y_str = ",".join(str(i) for i in y)
        dss.Text.Command(f"New {name} npts={npts} Yarray=({y_str}) Xarray=({x_str})")
        self._xycurve_counter += 1
        return name

    def set_properties_to_inverter(self, inverter: str, inv_control: InverterControl):
        """Modify a legacy InvControl object."""
        if inv_control.vvcontrol is not None:
            vvc_curve = self.create_xy_curve(
                inv_control.vvcontrol.voltage, inv_control.vvcontrol.reactive_response
            )
            dss.Text.Command(f"{inverter}.vvc_curve1={vvc_curve.split('.')[1]}")
            dss.Text.Command(
                f"{inverter}.deltaQ_factor={inv_control.vvcontrol.deltaq_factor}"
            )
            dss.Text.Command(
                f"{inverter}.VarChangeTolerance={inv_control.vvcontrol.varchangetolerance}"
            )
            dss.Text.Command(
                f"{inverter}.VoltageChangeTolerance={inv_control.vvcontrol.voltagechangetolerance}"
            )
            dss.Text.Command(
                f"{inverter}.VV_RefReactivePower={inv_control.vvcontrol.vv_refreactivepower}"
            )
        if inv_control.vwcontrol is not None:
            vw_curve = self.create_xy_curve(
                inv_control.vwcontrol.voltage,
                inv_control.vwcontrol.power_response,
            )
            dss.Text.Command(f"{inverter}.voltwatt_curve={vw_curve.split('.')[1]}")
            dss.Text.Command(
                f"{inverter}.deltaP_factor={inv_control.vwcontrol.deltap_factor}"
            )
        if inv_control.mode == InverterControlMode.voltvar_voltwatt:
            dss.Text.Command(f"{inverter}.CombiMode = VV_VW")
        else:
            dss.Text.Command(f"{inverter}.Mode = {inv_control.mode.value}")

    def set_pv_output(self, pv_system, p, q):
        """Set P and Q output for a PV system in OpenDSS."""
        max_pv = self.get_max_pv_available(pv_system)

        obj_name = f"PVSystem.{pv_system}"
        if max_pv <= 0 or p == 0:
            Warning("Maximum PV Value is 0")
            obj_val = 100
            q = 0
        elif p < max_pv:
            obj_val = p / float(max_pv) * 100
        else:
            obj_val = 100
            ratio = float(max_pv) / p
            q = q * ratio
        command = [
            Command(obj_name=obj_name, obj_property="%Pmpp", val=str(obj_val)),
            Command(obj_name=obj_name, obj_property="kvar", val=str(q)),
            Command(obj_name=obj_name, obj_property="%Cutout", val="0"),
            Command(obj_name=obj_name, obj_property="%Cutin", val="0"),
        ]
        self.change_obj(command)

    def get_max_pv_available(self, pv_system):
        irradiance = None
        pmpp = None
        flag = dss.PVsystems.First()
        while flag:
            if dss.PVsystems.Name() == pv_system:
                irradiance = dss.PVsystems.IrradianceNow()
                pmpp = dss.PVsystems.Pmpp()
            flag = dss.PVsystems.Next()
        if irradiance is None or pmpp is None:
            raise ValueError(f"Irradiance or PMPP not found for {pv_system}")
        return irradiance * pmpp

    def get_available_pv(self):
        pv_names = []
        powers = []
        flag = dss.PVsystems.First()
        while flag:
            pv_names.append(f"PVSystem.{dss.PVsystems.Name()}")
            powers.append(dss.PVsystems.Pmpp() * dss.PVsystems.IrradianceNow())
            flag = dss.PVsystems.Next()
        return xr.DataArray(powers, coords={"ids": pv_names})

    def apply_inverter_control(self, inv_control: InverterControl):
        """Apply inverter control, creating InvControl object if necessary."""
        if inv_control.pvsystem_list is None:
            pvsystem_set = self._pvsystems
        else:
            pvsystem_set = set(inv_control.pvsystem_list)
        inverter_set = set(
            self._pvsystem_to_inverter[pvsystem]
            for pvsystem in pvsystem_set
            if pvsystem in self._pvsystem_to_inverter
        )
        if len(inverter_set) == 1:
            (inverter,) = inverter_set
        else:
            inverter = self.create_inverter(pvsystem_set)

        assert (
            self._inverter_to_pvsystems[inverter] == pvsystem_set
        ), f"{self._inverter_to_pvsystems[inverter]} does not match {pvsystem_set} for {inverter}"

        self.set_properties_to_inverter(inverter, inv_control)
        return inverter

    def get_incidences(self) -> IncidenceList:
        """Get incidence list from line names to buses."""
        assert self._state != OpenDSSState.UNLOADED, f"{self._state}"
        from_list = []
        to_list = []
        equipment_ids = []
        equipment_types = []
        for line in dss.Lines.AllNames():
            dss.Circuit.SetActiveElement("Line." + line)
            names = dss.CktElement.BusNames()
            if len(names) != 2:
                bus_names = map(lambda x: x.split(".")[0], names)
                names = list(dict.fromkeys(bus_names))
                if len(names) != 2:
                    logging.info(
                        f"Line {line} has {len(names)} terminals, skipping in incidence matrix"
                    )
                    continue
            from_bus, to_bus = names
            from_list.append(from_bus.upper())
            to_list.append(to_bus.upper())
            equipment_ids.append(line)
            equipment_types.append("Line")
        for transformer in dss.Transformers.AllNames():
            dss.Circuit.SetActiveElement("Transformer." + transformer)
            names = dss.CktElement.BusNames()
            if len(names) != 2:
                bus_names = map(lambda x: x.split(".")[0], names)
                names = list(dict.fromkeys(bus_names))
                if len(names) != 2:
                    logging.info(
                        f"Transformer {transformer} has {len(names)} terminals, skipping in incidence matrix"
                    )
                    continue
            from_bus, to_bus = names
            from_list.append(from_bus.upper())
            to_list.append(to_bus.upper())
            equipment_ids.append(transformer)
            equipment_types.append("Transformer")
        return IncidenceList(
            from_equipment=from_list,
            to_equipment=to_list,
            ids=equipment_ids,
            equipment_types=equipment_types,
        )
