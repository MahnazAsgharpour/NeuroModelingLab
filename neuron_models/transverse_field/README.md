# Transverse Electric-Field Threshold Model

This directory contains a NEURON-based simulation for investigating the
effects of transverse electric fields on neuronal compartments.

The model represents each longitudinal compartment using a central section
connected to multiple radial spoke compartments. Extracellular potentials
are assigned according to the position of each spoke relative to the applied
electric field.

The simulation estimates the minimum electric-field amplitude required to
evoke an action potential for different compartment radii and stimulation
pulse widths.

## Current implementation

- NEURON Python interface
- Spoke–hub transverse morphology
- Extracellular stimulation
- Action-potential detection
- Iterative threshold search
- Configurable pulse width and compartment radius

This implementation was developed as part of research on TMS-induced
polarization and neuronal activation.
