## BBP model loader

The `bbp_loader.py` module loads Blue Brain Project cell models into
NEURON by reading their morphology, biophysics, synapse, and template
HOC files.

The loader:

- validates the expected model files;
- identifies the NEURON template;
- imports ASC morphology files;
- optionally loads synaptic mechanisms;
- supports axon replacement where provided by the original model;
- returns an instantiated NEURON cell object.

The original BBP model files remain subject to their original licenses
and attribution requirements.
