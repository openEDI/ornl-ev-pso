# OEDISI EVCS - IEEE 123-bus 

EV Charging  use case for OEDISI

## Simulation Parameters

| Parameter | Value |
|-----------|-------|
| Feeder | IEEE 123-bus |
| Time resolution | 1 hour |
| Duration | 24 hours (24 timesteps) |
| EVs | 40 (15 + 12 + 13 across 3 stations) |
| EVCS buses | 48.1, 65.1, 76.1 |
| Battery capacity | 50 kWh |
| PSO | 30 particles, 30 iterations |

## Data Flow

```
feeder ──powers_real──────→ evcs
feeder ──powers_imag──────→ evcs
feeder ──topology─────────→ evcs
feeder ──voltages_real────→ evcs
feeder ──voltages_imag────→ evcs
evcs   ──ev_load_real─────→ feeder
evcs   ──ev_load_imag─────→ feeder
feeder ──powers_real──────→ recorder_power_real
feeder ──powers_imag──────→ recorder_power_imag
feeder ──voltages_magnitude→ recorder_voltage_magnitude
```