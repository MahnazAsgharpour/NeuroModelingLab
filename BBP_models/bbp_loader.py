"""Utilities for loading Blue Brain Project cell models in NEURON."""

from pathlib import Path
from typing import Any

__all__ = ["build_bbp_cell"]


def build_bbp_cell(
    model_dir: str | Path,
    add_synapses: bool = False,
    replace_axon: bool = False,
    nrn: Any | None = None,
    model_name: str | None = None,
    verbose: bool = False,
):
    """
    Build a Blue Brain Project cell model in NEURON.

    Parameters
    ----------
    model_dir
        Directory containing the BBP cell-model files.
    add_synapses
        Whether synaptic mechanisms should be loaded and instantiated.
    replace_axon
        Whether the model constructor should replace the original axon.
    nrn
        Optional NEURON module or wrapper exposing an ``h`` attribute.
    model_name
        Optional descriptive name used in messages.
    verbose
        Print information about the detected template.

    Returns
    -------
    hoc.HocObject
        Instantiated NEURON cell model.

    Raises
    ------
    FileNotFoundError
        If required model files or morphology files are missing.
    ValueError
        If no template declaration can be found.
    """
    model_path = Path(model_dir).expanduser().resolve()

    if not model_path.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {model_path}")

    if model_name is None:
        model_name = model_path.name

    hoc_files = {
        "morphology": model_path / "morphology_2.hoc",
        "biophysics": model_path / "biophysics.hoc",
        "template": model_path / "template_2.hoc",
    }

    if add_synapses:
        hoc_files["synapses"] = model_path / "synapses" / "synapses.hoc"

    for file_type, file_path in hoc_files.items():
        if not file_path.is_file():
            raise FileNotFoundError(
                f"Required {file_type} file not found for "
                f"'{model_name}': {file_path}"
            )

    morphology_files = list((model_path / "morphology").glob("*.asc"))

    if not morphology_files:
        raise FileNotFoundError(
            f"No ASC morphology file found in: "
            f"{model_path / 'morphology'}"
        )

    if len(morphology_files) > 1:
        raise ValueError(
            f"Multiple ASC morphology files found for '{model_name}'. "
            "Specify or retain only the intended morphology file."
        )

    morphology_file = morphology_files[0]

    template_name = None

    with hoc_files["template"].open("r", encoding="utf-8") as template_file:
        for line in template_file:
            stripped_line = line.strip()

            if stripped_line.startswith("begintemplate"):
                parts = stripped_line.split()

                if len(parts) >= 2:
                    template_name = parts[-1]
                    break

    if template_name is None:
        raise ValueError(
            f"No 'begintemplate' declaration found in "
            f"{hoc_files['template']}"
        )

    if verbose:
        print(f"Loading BBP model: {model_name}")
        print(f"Template name: {template_name}")
        print(f"Morphology: {morphology_file.name}")

    if nrn is not None:
        h = nrn.h
    else:
        from neuron import h

    h.load_file("import3d.hoc")
    h.load_file(str(hoc_files["morphology"]))
    h.load_file(str(hoc_files["biophysics"]))

    if add_synapses:
        h.load_file(str(hoc_files["synapses"]))

    h.load_file(str(hoc_files["template"]))

    try:
        cell_template = getattr(h, template_name)
    except AttributeError as exc:
        raise RuntimeError(
            f"NEURON template '{template_name}' was not created after "
            f"loading {hoc_files['template']}."
        ) from exc

    return cell_template(
        int(add_synapses),
        int(replace_axon),
        str(morphology_file),
    )
