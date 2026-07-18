"""
Passive cable response to intracellular and extracellular stimulation.

This script creates a passive neuronal cable in NEURON, defines its
three-dimensional morphology, applies a time-varying uniform extracellular
electric field, and records membrane responses at several cable locations.

The example is intended as a transparent demonstration of:

1. Passive cable construction and discretization.
2. Three-dimensional segment-coordinate interpolation.
3. Conversion of a uniform electric field into extracellular potential.
4. Application of extracellular potentials with NEURON Vector.play().
5. Comparison of voltage responses along the cable.

This is a passive model and therefore is not intended to generate action
potentials or calculate activation thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from neuron import h


h.load_file("stdrun.hoc")


@dataclass(frozen=True)
class CableParameters:
    """Electrical and geometric parameters of the passive cable."""

    length_um: float = 800.0
    diameter_um: float = 5.0
    membrane_capacitance_uf_cm2: float = 1.0
    axial_resistivity_ohm_cm: float = 100.0
    membrane_resistance_ohm_cm2: float = 15_000.0
    resting_potential_mv: float = -65.0
    lambda_frequency_hz: float = 100.0


@dataclass(frozen=True)
class SimulationParameters:
    """Numerical and stimulation settings."""

    dt_ms: float = 0.025
    tstop_ms: float = 50.0
    temperature_c: float = 34.0

    current_clamp_location: float = 0.5
    current_clamp_delay_ms: float = 5.0
    current_clamp_duration_ms: float = 30.0
    current_clamp_amplitude_na: float = 0.1

    field_onset_ms: float = 5.0
    field_duration_ms: float = 30.0
    field_magnitude_mv_per_um: float = 0.001
    theta_deg: float = 180.0
    phi_deg: float = 0.0


def odd_nseg_from_dlambda(
    section: h.Section,
    frequency_hz: float = 100.0,
    d_lambda: float = 0.1,
) -> int:
    """
    Calculate an odd number of segments using NEURON's d-lambda rule.
    """
    lambda_um = h.lambda_f(frequency_hz, sec=section)
    nseg = int((section.L / (d_lambda * lambda_um) + 0.9) / 2.0) * 2 + 1
    return max(1, nseg)


def create_passive_cable(params: CableParameters) -> h.Section:
    """
    Create a straight passive cable aligned with the positive z-axis.
    """
    cable = h.Section(name="passive_cable")

    cable.pt3dclear()
    cable.pt3dadd(0.0, 0.0, 0.0, params.diameter_um)
    cable.pt3dadd(
        0.0,
        0.0,
        params.length_um,
        params.diameter_um,
    )

    cable.L = params.length_um
    cable.diam = params.diameter_um
    cable.cm = params.membrane_capacitance_uf_cm2
    cable.Ra = params.axial_resistivity_ohm_cm

    cable.insert("pas")
    cable.g_pas = 1.0 / params.membrane_resistance_ohm_cm2
    cable.e_pas = params.resting_potential_mv

    cable.insert("extracellular")
    cable.e_extracellular = 0.0

    cable.nseg = odd_nseg_from_dlambda(
        cable,
        frequency_hz=params.lambda_frequency_hz,
    )

    h.define_shape()
    return cable


def interpolate_3d_coordinates(
    section: h.Section,
    normalized_arc: float,
) -> tuple[float, float, float]:
    """
    Interpolate the 3D coordinates at a normalized arc position.

    Parameters
    ----------
    section
        NEURON section containing pt3d morphology points.
    normalized_arc
        Relative position along the section in the interval [0, 1].
    """
    if not 0.0 <= normalized_arc <= 1.0:
        raise ValueError("normalized_arc must be between 0 and 1.")

    n3d = int(section.n3d())

    if n3d < 2:
        raise ValueError(
            f"Section '{section.name()}' does not have sufficient 3D points."
        )

    target_arc = normalized_arc * section.arc3d(n3d - 1)

    for index in range(n3d - 1):
        arc_start = section.arc3d(index)
        arc_end = section.arc3d(index + 1)

        if arc_start <= target_arc <= arc_end:
            interval = arc_end - arc_start

            if interval == 0:
                fraction = 0.0
            else:
                fraction = (target_arc - arc_start) / interval

            x = (
                (1.0 - fraction) * section.x3d(index)
                + fraction * section.x3d(index + 1)
            )
            y = (
                (1.0 - fraction) * section.y3d(index)
                + fraction * section.y3d(index + 1)
            )
            z = (
                (1.0 - fraction) * section.z3d(index)
                + fraction * section.z3d(index + 1)
            )
            return float(x), float(y), float(z)

    return (
        float(section.x3d(n3d - 1)),
        float(section.y3d(n3d - 1)),
        float(section.z3d(n3d - 1)),
    )


def electric_field_components(
    magnitude_mv_per_um: float,
    theta_deg: float,
    phi_deg: float,
) -> np.ndarray:
    """
    Convert spherical field parameters into Cartesian components.

    theta is the polar angle measured from the positive z-axis.
    phi is the azimuthal angle measured in the x-y plane.
    """
    theta = np.radians(theta_deg)
    phi = np.radians(phi_deg)

    return np.array(
        [
            magnitude_mv_per_um * np.sin(theta) * np.cos(phi),
            magnitude_mv_per_um * np.sin(theta) * np.sin(phi),
            magnitude_mv_per_um * np.cos(theta),
        ],
        dtype=float,
    )


def create_field_waveform(
    simulation: SimulationParameters,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create the simulation time vector and a square field-pulse waveform.
    """
    time_ms = np.arange(
        0.0,
        simulation.tstop_ms + simulation.dt_ms / 2.0,
        simulation.dt_ms,
    )

    field_scale = np.zeros_like(time_ms)
    field_end_ms = simulation.field_onset_ms + simulation.field_duration_ms

    pulse_mask = (
        (time_ms >= simulation.field_onset_ms)
        & (time_ms < field_end_ms)
    )
    field_scale[pulse_mask] = 1.0

    return time_ms, field_scale


def extracellular_potential_matrix(
    section: h.Section,
    time_ms: np.ndarray,
    field_scale: np.ndarray,
    field_vector_mv_per_um: np.ndarray,
) -> np.ndarray:
    """
    Compute extracellular potential at every segment over time.

    The potential is calculated from

        Ve = -(E dot r),

    where E is expressed in mV/um and r in um, giving Ve in mV.
    """
    if time_ms.shape != field_scale.shape:
        raise ValueError("time_ms and field_scale must have equal shapes.")

    segment_positions = []

    for segment in section:
        position = interpolate_3d_coordinates(section, float(segment.x))
        segment_positions.append(position)

    positions_um = np.asarray(segment_positions, dtype=float)
    static_potential_mv = -(positions_um @ field_vector_mv_per_um)

    return np.outer(field_scale, static_potential_mv)


def play_extracellular_potentials(
    section: h.Section,
    time_ms: np.ndarray,
    potentials_mv: np.ndarray,
) -> tuple[h.Vector, list[h.Vector]]:
    """
    Play one extracellular-potential waveform into each cable segment.

    Returned vectors must remain alive until the simulation is complete.
    """
    if potentials_mv.shape != (len(time_ms), section.nseg):
        raise ValueError(
            "potentials_mv must have shape "
            f"({len(time_ms)}, {section.nseg})."
        )

    time_vector = h.Vector(time_ms)
    played_vectors: list[h.Vector] = []

    for segment_index, segment in enumerate(section):
        potential_vector = h.Vector(potentials_mv[:, segment_index])
        potential_vector.play(
            segment._ref_e_extracellular,
            time_vector,
            1,
        )
        played_vectors.append(potential_vector)

    return time_vector, played_vectors


def configure_current_clamp(
    cable: h.Section,
    simulation: SimulationParameters,
) -> h.IClamp:
    """
    Add an intracellular current clamp to the cable.
    """
    clamp = h.IClamp(cable(simulation.current_clamp_location))
    clamp.delay = simulation.current_clamp_delay_ms
    clamp.dur = simulation.current_clamp_duration_ms
    clamp.amp = simulation.current_clamp_amplitude_na
    return clamp


def record_voltage(
    cable: h.Section,
    locations: Iterable[float],
) -> dict[float, h.Vector]:
    """
    Record membrane potential at selected normalized cable locations.
    """
    recordings: dict[float, h.Vector] = {}

    for location in locations:
        if not 0.0 <= location <= 1.0:
            raise ValueError("Recording locations must be between 0 and 1.")

        recordings[location] = h.Vector().record(cable(location)._ref_v)

    return recordings


def run_simulation(
    cable_params: CableParameters,
    simulation_params: SimulationParameters,
) -> dict[str, object]:
    """
    Build the model, apply stimulation, and execute the simulation.
    """
    cable = create_passive_cable(cable_params)
    current_clamp = configure_current_clamp(cable, simulation_params)

    field_vector = electric_field_components(
        simulation_params.field_magnitude_mv_per_um,
        simulation_params.theta_deg,
        simulation_params.phi_deg,
    )

    field_time_ms, field_scale = create_field_waveform(simulation_params)
    potentials_mv = extracellular_potential_matrix(
        cable,
        field_time_ms,
        field_scale,
        field_vector,
    )

    playback_time, playback_vectors = play_extracellular_potentials(
        cable,
        field_time_ms,
        potentials_mv,
    )

    recording_locations = (0.1, 0.5, 0.9)
    voltage_recordings = record_voltage(cable, recording_locations)
    simulation_time = h.Vector().record(h._ref_t)

    h.dt = simulation_params.dt_ms
    h.steps_per_ms = 1.0 / simulation_params.dt_ms
    h.tstop = simulation_params.tstop_ms
    h.celsius = simulation_params.temperature_c
    h.v_init = cable_params.resting_potential_mv

    h.finitialize(cable_params.resting_potential_mv)
    h.continuerun(simulation_params.tstop_ms)

    return {
        "cable": cable,
        "current_clamp": current_clamp,
        "field_vector": field_vector,
        "field_time_ms": field_time_ms,
        "field_scale": field_scale,
        "extracellular_potentials_mv": potentials_mv,
        "playback_time": playback_time,
        "playback_vectors": playback_vectors,
        "time_ms": np.asarray(simulation_time),
        "voltage_recordings_mv": {
            location: np.asarray(vector)
            for location, vector in voltage_recordings.items()
        },
    }


def plot_results(
    results: dict[str, object],
    simulation_params: SimulationParameters,
    output_path: str | Path | None = None,
) -> None:
    """
    Plot applied field waveform and membrane responses.
    """
    time_ms = np.asarray(results["time_ms"])
    voltage_recordings = results["voltage_recordings_mv"]
    field_time_ms = np.asarray(results["field_time_ms"])
    field_scale = np.asarray(results["field_scale"])

    figure = plt.figure(figsize=(10, 8))

    voltage_axis = figure.add_subplot(2, 1, 1)
    for location, voltage_mv in voltage_recordings.items():
        voltage_axis.plot(
            time_ms,
            voltage_mv,
            label=f"x = {location:.1f}",
        )

    voltage_axis.set_xlabel("Time (ms)")
    voltage_axis.set_ylabel("Membrane potential (mV)")
    voltage_axis.set_title("Passive cable membrane response")
    voltage_axis.grid(True)
    voltage_axis.legend()

    field_axis = figure.add_subplot(2, 1, 2)
    field_axis.plot(
        field_time_ms,
        field_scale * simulation_params.field_magnitude_mv_per_um,
    )
    field_axis.set_xlabel("Time (ms)")
    field_axis.set_ylabel("Field magnitude (mV/um)")
    field_axis.set_title("Applied extracellular electric-field pulse")
    field_axis.grid(True)

    figure.tight_layout()

    if output_path is not None:
        figure.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.show()


def main() -> None:
    """Run the passive cable demonstration."""
    cable_params = CableParameters()
    simulation_params = SimulationParameters()

    results = run_simulation(cable_params, simulation_params)

    cable = results["cable"]
    field_vector = np.asarray(results["field_vector"])

    print(
        f"Cable length: {cable.L:.1f} um\n"
        f"Cable diameter: {cable.diam:.1f} um\n"
        f"Number of segments: {cable.nseg}\n"
        f"Electric-field vector: "
        f"({field_vector[0]:.6f}, "
        f"{field_vector[1]:.6f}, "
        f"{field_vector[2]:.6f}) mV/um"
    )

    plot_results(
        results,
        simulation_params,
        output_path="passive_cable_field_response.png",
    )


if __name__ == "__main__":
    main()
