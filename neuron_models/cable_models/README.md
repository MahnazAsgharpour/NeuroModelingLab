# Passive Cable Electric-Field Demonstration

This directory contains a NEURON-based demonstration of the response of a
passive neuronal cable to intracellular current injection and a uniform
extracellular electric field.

The original exploratory script has been reorganized into reusable functions
with explicit units, input validation, three-dimensional morphology, and
correct application of extracellular potential through NEURON's
`extracellular` mechanism.

## Scientific purpose

The model illustrates how the orientation and magnitude of an external
electric field can produce spatially varying extracellular potentials along
a neuronal cable.

For a uniform field, the extracellular potential at position **r** is computed
as:

```text
Ve = -(E · r)
```

where:

- `E` is the electric-field vector in mV/µm;
- `r` is the segment position in µm;
- `Ve` is the extracellular potential in mV.

The calculated potential waveform is applied independently to every segment
using `Vector.play()` and `e_extracellular`.

## Model features

- Straight passive cable with explicit 3D morphology
- Passive membrane mechanism (`pas`)
- NEURON `extracellular` mechanism
- d-lambda spatial discretization
- Configurable intracellular current clamp
- Configurable electric-field magnitude and direction
- Square extracellular field pulse
- Voltage recordings at multiple cable locations
- Plotting of the field waveform and membrane responses

## Important limitation

This is a **passive cable model**. It does not contain voltage-gated sodium or
potassium channels and therefore is not intended to:

- generate action potentials;
- estimate activation thresholds;
- represent a complete axon or Purkinje-cell model;
- provide a validated model of TMS activation.

It should be treated as a methodological and educational example for
extracellular-field application.

## File

```text
passive_cable_field_demo.py
```

## Requirements

- Python 3.10 or newer
- NEURON
- NumPy
- Matplotlib

Install the required Python packages with:

```bash
pip install neuron numpy matplotlib
```

## Running the example

From this directory, run:

```bash
python passive_cable_field_demo.py
```

The script prints the cable geometry and electric-field components, opens a
figure showing the simulation results, and saves:

```text
passive_cable_field_response.png
```

## Main parameters

The model settings are defined in two dataclasses:

### `CableParameters`

Controls:

- cable length and diameter;
- membrane capacitance;
- axial resistivity;
- membrane resistance;
- passive reversal potential;
- d-lambda discretization frequency.

### `SimulationParameters`

Controls:

- integration time step and duration;
- temperature;
- current-clamp position, timing, and amplitude;
- electric-field pulse timing;
- electric-field magnitude;
- polar and azimuthal field angles.

For example:

```python
simulation_params = SimulationParameters(
    field_magnitude_mv_per_um=0.001,
    theta_deg=180.0,
    phi_deg=0.0,
)
```

The cable is aligned with the positive z-axis. Therefore:

- `theta_deg = 0` produces a field parallel to the positive z-axis;
- `theta_deg = 180` produces a field parallel to the negative z-axis;
- `theta_deg = 90` produces a field in the x-y plane.

## Code organization

The script is divided into functions for:

1. constructing the passive cable;
2. determining spatial discretization;
3. interpolating segment coordinates;
4. calculating electric-field components;
5. generating the field waveform;
6. calculating extracellular potentials;
7. applying potentials with `Vector.play()`;
8. recording membrane voltages;
9. running and plotting the simulation.

## Suggested repository location

```text
NeuroModelingLab/
└── neuron_models/
    └── cable_models/
        ├── passive_cable_field_demo.py
        └── README.md
```

## Attribution

This implementation is part of the NeuroModelingLab scientific software
collection for computational neuroscience, extracellular stimulation, and
neural modeling.
