"""PSO optimization for EV charging scheduling."""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def uncontrolled_charging(
    initial_soc,
    num_control_steps,
    control_interval,
    battery_capacity,
    charging_efficiency,
    arrival_time_idx,
    departure_time_idx,
    num_evs,
    max_charging_rate,
    desired_state_of_charge,
):
    """Charge each EV at max rate from arrival until target SOC is reached."""
    soc = np.zeros((num_evs, num_control_steps))
    charging_rate = np.zeros((num_evs, num_control_steps))
    soc_tolerance = 0.001

    for ev in range(num_evs):
        if arrival_time_idx[ev] < num_control_steps:
            soc[ev, arrival_time_idx[ev]] = initial_soc[ev]
            if arrival_time_idx[ev] > 0:
                soc[ev, 0 : arrival_time_idx[ev]] = initial_soc[ev]

    for t in range(num_control_steps - 1):
        for ev in range(num_evs):
            is_present = t >= arrival_time_idx[ev] and t < departure_time_idx[ev]
            needs_charge = soc[ev, t] < desired_state_of_charge - soc_tolerance

            if is_present and needs_charge:
                charging_rate[ev, t] = max_charging_rate
            else:
                charging_rate[ev, t] = 0

            if is_present:
                charged_energy = (
                    charging_rate[ev, t] * control_interval * charging_efficiency
                )
                soc[ev, t + 1] = soc[ev, t] + (charged_energy / battery_capacity)
            else:
                soc[ev, t + 1] = soc[ev, t]

            soc[ev, t + 1] = np.clip(soc[ev, t + 1], 0, desired_state_of_charge)

    charging_rate[:, -1] = 0
    return soc, charging_rate


def calculate_soc(
    initial_soc,
    charging_rate,
    num_control_steps,
    control_interval,
    battery_capacity,
    charging_efficiency,
    arrival_time_idx,
    departure_time_idx,
    num_evs,
):
    """Calculate SOC for each EV over all time steps."""
    soc = np.zeros((num_evs, num_control_steps))

    for ev in range(num_evs):
        if arrival_time_idx[ev] < num_control_steps:
            soc[ev, arrival_time_idx[ev]] = initial_soc[ev]
            if arrival_time_idx[ev] > 0:
                soc[ev, 0 : arrival_time_idx[ev]] = initial_soc[ev]

    for t in range(num_control_steps - 1):
        for ev in range(num_evs):
            if t >= arrival_time_idx[ev] and t < departure_time_idx[ev]:
                charged_energy = (
                    charging_rate[ev, t] * control_interval * charging_efficiency
                )
                soc[ev, t + 1] = soc[ev, t] + (charged_energy / battery_capacity)
            else:
                soc[ev, t + 1] = soc[ev, t]
            soc[ev, t + 1] = np.clip(soc[ev, t + 1], 0, 1)

    return soc


def simulate_real_charging_process(
    initial_soc,
    scheduled_charging_rate,
    num_control_steps,
    control_interval,
    battery_capacity,
    charging_efficiency,
    arrival_time_idx,
    departure_time_idx,
    num_evs,
):
    """Simulate battery physics: clamp actual rate when SOC would exceed 100%."""
    soc = np.zeros((num_evs, num_control_steps))
    real_charging_rate = np.zeros((num_evs, num_control_steps))

    for ev in range(num_evs):
        if arrival_time_idx[ev] < num_control_steps:
            soc[ev, : arrival_time_idx[ev] + 1] = initial_soc[ev]

    for t in range(num_control_steps - 1):
        for ev in range(num_evs):
            if t >= arrival_time_idx[ev] and t < departure_time_idx[ev]:
                current_soc = soc[ev, t]
                max_energy_input = (1.0 - current_soc) * battery_capacity
                attempted_energy = (
                    scheduled_charging_rate[ev, t]
                    * control_interval
                    * charging_efficiency
                )
                actual_energy = max(0, min(attempted_energy, max_energy_input))

                if actual_energy > 0:
                    real_charging_rate[ev, t] = actual_energy / (
                        control_interval * charging_efficiency
                    )
                else:
                    real_charging_rate[ev, t] = 0

                soc[ev, t + 1] = current_soc + (actual_energy / battery_capacity)
            else:
                real_charging_rate[ev, t] = 0
                soc[ev, t + 1] = soc[ev, t]

            soc[ev, t + 1] = np.clip(soc[ev, t + 1], 0, 1.0)

    return soc, real_charging_rate


def calculate_cost(
    charging_rate,
    electricity_price,
    num_control_steps,
    control_interval,
    num_evs,
    arrival_time_idx,
    departure_time_idx,
):
    """Calculate total charging cost."""
    total_cost = 0
    for t in range(num_control_steps):
        for ev in range(num_evs):
            if t >= arrival_time_idx[ev] and t < departure_time_idx[ev]:
                total_cost += (
                    charging_rate[ev, t] * control_interval * electricity_price[t]
                )
    return total_cost


def calculate_cost_per_step(
    charging_rate,
    electricity_price,
    num_control_steps,
    control_interval,
    num_evs,
    arrival_time_idx,
    departure_time_idx,
):
    """Calculate charging cost at each time step across all EVs."""
    cost_per_step = np.zeros(num_control_steps)
    for t in range(num_control_steps):
        step_cost = 0
        for ev in range(num_evs):
            if t >= arrival_time_idx[ev] and t < departure_time_idx[ev]:
                step_cost += (
                    charging_rate[ev, t] * control_interval * electricity_price[t]
                )
        cost_per_step[t] = step_cost
    return cost_per_step


def fitness_function(charging_rate, network, base_voltages, evcs_bus, ev_params):
    """Fitness = electricity cost + SOC undershoot penalty + linearized voltage penalty."""
    if ev_params is None:
        raise ValueError("ev_params is required")

    num_evs = ev_params["num_evs"]
    num_control_steps = ev_params["num_control_steps"]
    control_interval = ev_params["control_interval"]
    battery_capacity = ev_params["battery_capacity"]
    charging_efficiency = ev_params["charging_efficiency"]
    electricity_price = ev_params["electricity_price"]
    arrival_time_idx = ev_params["arrival_time_idx"]
    departure_time_idx = ev_params["departure_time_idx"]
    initial_soc = ev_params["initial_soc"]
    desired_state_of_charge = ev_params["desired_soc"]
    evcs_bus_assignment = ev_params["evcs_bus_assignment"]

    real_soc, real_rate = simulate_real_charging_process(
        initial_soc,
        charging_rate,
        num_control_steps,
        control_interval,
        battery_capacity,
        charging_efficiency,
        arrival_time_idx,
        departure_time_idx,
        num_evs,
    )

    cost = calculate_cost(
        real_rate,
        electricity_price,
        num_control_steps,
        control_interval,
        num_evs,
        arrival_time_idx,
        departure_time_idx,
    )

    undershoot_penalty = 0
    penalty_weight = 2000  # Proportional penalty preserves gradient information
    for ev in range(num_evs):
        dep_check_idx = min(departure_time_idx[ev] - 1, num_control_steps - 1)
        if dep_check_idx >= 0:
            final_soc = real_soc[ev, dep_check_idx]
            if final_soc < desired_state_of_charge - 0.01:
                undershoot_penalty += penalty_weight * (
                    desired_state_of_charge - final_soc
                )

    voltage_penalty = 0
    v_min, v_max = 0.95, 1.05

    for t in range(num_control_steps):
        ev_loads_per_bus = {}
        for bus, ev_indices in evcs_bus_assignment.items():
            ev_loads_per_bus[bus] = (
                np.sum(real_rate[ev_indices, t]) if ev_indices else 0.0
            )

        if network is not None:
            estimated_voltages = network.estimate_voltages(
                base_voltages, ev_loads_per_bus
            )
            violations = np.maximum(0, v_min - estimated_voltages) + np.maximum(
                0, estimated_voltages - v_max
            )
            if np.any(violations > 0):
                voltage_penalty += 1000 * np.sum(violations)

    return cost + undershoot_penalty + voltage_penalty


def ev_pso_optimization(
    num_particles, max_iterations, network, base_voltages, evcs_bus, ev_params
):
    """PSO optimization using LinearizedNetwork for voltage estimation (~1000x faster than OpenDSS)."""
    if ev_params is None:
        raise ValueError("ev_params is required")

    num_evs = ev_params["num_evs"]
    max_charging_rate = ev_params["max_charging_rate"]
    charging_efficiency = ev_params["charging_efficiency"]
    battery_capacity = ev_params["battery_capacity"]
    control_interval = ev_params["control_interval"]
    num_control_steps = ev_params["num_control_steps"]
    electricity_price = ev_params["electricity_price"]
    arrival_time_idx = ev_params["arrival_time_idx"]
    departure_time_idx = ev_params["departure_time_idx"]
    initial_soc = ev_params["initial_soc"]
    desired_state_of_charge = ev_params["desired_soc"]
    charging_energy = ev_params["charging_energy"]

    logger.info("========== PSO INITIALIZATION ==========")
    logger.info(
        f"EVs: {num_evs}, Time steps: {num_control_steps}, Max rate: {max_charging_rate} kW"
    )
    logger.info(f"Target SOC: {desired_state_of_charge*100:.0f}%")
    logger.info(
        f"Initial SOC range: {np.min(initial_soc)*100:.1f}% - {np.max(initial_soc)*100:.1f}%"
    )
    logger.info(f"Particles: {num_particles}, Iterations: {max_iterations}")
    logger.info("Using LinearizedNetwork for fast voltage estimation")
    logger.info("Using Hot Start (Baseline Strategy)...")

    baseline_soc, baseline_rate = uncontrolled_charging(
        initial_soc,
        num_control_steps,
        control_interval,
        battery_capacity,
        charging_efficiency,
        arrival_time_idx,
        departure_time_idx,
        num_evs,
        max_charging_rate,
        desired_state_of_charge,
    )

    baseline_total_energy = np.sum(baseline_rate) * control_interval
    logger.info(f"Baseline total energy: {baseline_total_energy:.1f} kWh")

    particles = np.random.uniform(
        0, max_charging_rate, (num_particles, num_evs, num_control_steps)
    )
    particles[0] = baseline_rate.copy()

    # Seed nearby particles with noisy baseline for warm start
    num_seeded = min(10, num_particles - 1)
    for p in range(1, num_seeded + 1):
        noise = np.random.normal(0, max_charging_rate * 0.1, baseline_rate.shape)
        particles[p] = np.clip(baseline_rate + noise, 0, max_charging_rate)

    logger.info(f"Seeded {num_seeded + 1} particles with baseline strategy")

    for p in range(num_particles):
        for ev in range(num_evs):
            particles[p, ev, : arrival_time_idx[ev]] = 0
            particles[p, ev, departure_time_idx[ev] :] = 0

    # Velocity clipped to +/-20% of max rate
    velocities = np.random.uniform(
        -max_charging_rate * 0.2,
        max_charging_rate * 0.2,
        (num_particles, num_evs, num_control_steps),
    )

    logger.info("Evaluating initial particle fitnesses...")

    personal_best_positions = particles.copy()
    personal_best_fitnesses = np.array(
        [
            fitness_function(particles[i], network, base_voltages, evcs_bus, ev_params)
            for i in range(num_particles)
        ]
    )
    global_best_index = np.argmin(personal_best_fitnesses)
    global_best_position = personal_best_positions[global_best_index].copy()
    global_best_fitness = personal_best_fitnesses[global_best_index]

    logger.info(f"Baseline fitness (Particle 0): {personal_best_fitnesses[0]:.2f}")
    logger.info(
        f"Initial best fitness: {global_best_fitness:.2f} (Particle {global_best_index})"
    )
    logger.info("========== OPTIMIZATION STARTING ==========")

    inertia_weight = 0.8
    cognitive_coefficient = 1.4
    social_coefficient = 1.4
    decay = 0.99

    for iteration in range(max_iterations):
        for i in range(num_particles):
            r1 = np.random.rand(num_evs, num_control_steps)
            r2 = np.random.rand(num_evs, num_control_steps)
            velocities[i] = (
                inertia_weight * velocities[i]
                + cognitive_coefficient
                * r1
                * (personal_best_positions[i] - particles[i])
                + social_coefficient * r2 * (global_best_position - particles[i])
            )

            velocities[i] = np.clip(
                velocities[i], -max_charging_rate * 0.2, max_charging_rate * 0.2
            )
            particles[i] = particles[i] + velocities[i]

            for ev in range(num_evs):
                particles[i, ev, : arrival_time_idx[ev]] = 0
                particles[i, ev, departure_time_idx[ev] :] = 0
                particles[i, ev, arrival_time_idx[ev] : departure_time_idx[ev]] = (
                    np.clip(
                        particles[i, ev, arrival_time_idx[ev] : departure_time_idx[ev]],
                        0,
                        max_charging_rate,
                    )
                )

                # Truncate schedule once cumulative energy exceeds 105% of required energy
                cumulated_energy = 0
                for j in range(num_control_steps):
                    cumulated_energy = (
                        cumulated_energy + particles[i, ev, j] * control_interval
                    )
                    if cumulated_energy > 1.05 * charging_energy[ev]:
                        particles[i, ev, j + 1 :] = 0
                        break

            current_fitness = fitness_function(
                particles[i], network, base_voltages, evcs_bus, ev_params
            )

            if current_fitness < personal_best_fitnesses[i]:
                personal_best_positions[i] = particles[i].copy()
                personal_best_fitnesses[i] = current_fitness

                if current_fitness < global_best_fitness:
                    global_best_position = particles[i].copy()
                    global_best_fitness = current_fitness

        inertia_weight *= decay

        if (iteration + 1) % 5 == 0 or iteration == max_iterations - 1:
            logger.info(
                f"Iter {iteration + 1:2d}/{max_iterations}: Best={global_best_fitness:.2f}, Inertia={inertia_weight:.3f}"
            )

    logger.info("========== OPTIMIZATION COMPLETE ==========")
    logger.info(f"Final optimized fitness: {global_best_fitness:.2f}")

    final_soc, real_final_rates = simulate_real_charging_process(
        initial_soc,
        global_best_position,
        num_control_steps,
        control_interval,
        battery_capacity,
        charging_efficiency,
        arrival_time_idx,
        departure_time_idx,
        num_evs,
    )

    evs_at_target = 0
    for ev in range(num_evs):
        dep_idx = min(departure_time_idx[ev] - 1, num_control_steps - 1)
        if final_soc[ev, dep_idx] >= desired_state_of_charge - 0.01:
            evs_at_target += 1
    logger.info(f"EVs reaching target SOC: {evs_at_target}/{num_evs}")

    final_true_cost = calculate_cost(
        real_final_rates,
        electricity_price,
        num_control_steps,
        control_interval,
        num_evs,
        arrival_time_idx,
        departure_time_idx,
    )

    optimized_energy = np.sum(real_final_rates) * control_interval
    logger.info(f"Total energy scheduled: {optimized_energy:.1f} kWh")
    logger.info(f"True electricity cost: ${final_true_cost:.2f}")
    logger.info("============================================")

    return global_best_position, final_true_cost
