
import os
import glob
import numpy as np


__all__ = ['build_BBP_cell', 'compute_Ve_over_cell', 'compute_Ve_over_section']


def build_BBP_cell(model_dir, add_synapses, replace_axon, nrn=None, model_name=None, verbose=False):
    if model_name is None:
        model_name = os.path.basename(model_dir)
    hoc_files = {
        'morpho': os.path.join(model_dir, 'morphology_2.hoc'),
        'biophys': os.path.join(model_dir, 'biophysics.hoc'),
        'synapses': os.path.join(model_dir, 'synapses', 'synapses.hoc'),
        'template': os.path.join(model_dir, 'template_2.hoc')
    }
    for f in hoc_files.values():
        if not os.path.isfile(f):
            raise Exception(f"'{f}': no such file")
    morphology_file = glob.glob(os.path.join(model_dir, 'morphology', '*.asc'))[0]
    if not os.path.isfile(morphology_file):
        raise Exception(f"'{morphology_file}': no such file")

    with open(hoc_files['template'], 'r') as fid:
        for line in fid:
            if 'begintemplate' in line:
                template_name = line.rstrip().split(' ')[-1]
    if verbose:
        print(f"Template name: '{template_name}'.")

    if nrn is not None:
        h_ = nrn.h
    else:
        from neuron import h as h_
    h_.load_file('import3d.hoc')
    h_.load_file(hoc_files['morpho'])
    h_.load_file(hoc_files['biophys'])
    h_.load_file(hoc_files['synapses'])
    h_.load_file(hoc_files['template'])

    # the constructor
    init_template = getattr(h_, template_name)
    return init_template(add_synapses, replace_axon, morphology_file)


def compute_Ve_over_cell(cell, Efield, full_output=False):
    # potential difference between the parent and the present points
    _compute_Ve = lambda x_c, x_p, Ve_p, Efun: Ve_p - 0.5*(Efun(x_c)+Efun(x_p))@((x_c-x_p)*1e-6)
    # the relative locations of all segments in a section
    _segment_loc = lambda nseg: np.linspace(0, 1, 2*nseg+1)[1::2]

    from patch import objects
    from neuron import nrn
    if isinstance(cell.soma[0], nrn.Section) and hasattr(cell, 'all'):
        # potentially a cell from the repo of the Blue Brain Project
        cell_secs = cell.all
    elif isinstance(cell.soma[0], objects.Section) and hasattr(cell, 'soma') \
         and hasattr(cell, 'dendrites') and hasattr(cell, 'axon'):
        # potentially a cell from the repo of the Brain and Behavioral
        # Sciences Laboratory at the University of Pavia
        cell_secs = cell.soma + cell.dendrites + cell.axon
    else:
        raise Exception('Do not know how to deal with this type of cell')

    points = {sec.name(): np.array([[sec.x3d(i), # x coord
                                     sec.y3d(i), # y coord
                                     sec.z3d(i), # z coord
                                     sec.arc3d(i)/sec.L, # relative position in section
                                     # electrical potential (will be filled later)
                                     0] for i in range(sec.n3d())]) for sec in cell_secs}
    segments = {sec.name(): np.zeros(sec.nseg) for sec in cell_secs}

    for sec in cell_secs:
        key_c = sec.name()
        if sec.parentseg() is not None:
            X_c = points[key_c][0][:3]
            key_p = sec.parentseg().sec.name()
            if np.sum((X_c - points[key_p][-1][:3])**2) < 1e-6:
                Ve = points[key_p][-1][-1]
            else:
                X_p = points[key_p][-2][:3]
                assert np.sum((X_c - X_p)**2) > 0
                Ve = _compute_Ve(X_c, X_p, points[key_p][-2][-1], Efield)
            points[key_c][0][-1] = Ve
        for j in range(1, sec.n3d()):
            X_c = points[key_c][j][:3]
            X_p = points[key_c][j-1][:3]
            points[key_c][j][-1] = _compute_Ve(X_c, X_p, points[key_c][j-1][-1], Efield)

        bins = np.linspace(0, 1, sec.nseg+1)
        pos = points[key_c][:,3]
        idx = np.digitize(pos, bins, right=False) - 1
        idx[idx == sec.nseg] = sec.nseg-1
        for j in range(sec.nseg):
            jdx, = np.where(idx == j)
            if jdx.size == 0:
                if j == 0:
                    Ve = segments[key_p][-1]
                else:
                    Ve = segments[key_c][j-1]
            else:
                Ve = points[key_c][jdx,-1].mean()
            segments[key_c][j] = Ve
    V_extra = [segments[sec.name()] for sec in cell_secs]
    if full_output:
        return V_extra,cell_secs,points
    return V_extra


def compute_Ve_over_section(sec, E, theta, phi):
    fun = lambda XYZ,E,theta,phi: -abs(E)*(XYZ[0]*np.sin(theta)*np.cos(phi) + \
                                           XYZ[1]*np.sin(theta)*np.sin(phi) + \
                                           XYZ[2]*np.cos(theta))
    n_pts = sec.n3d()
    bins = np.linspace(0, 1, sec.nseg+1)
    pos = np.array([sec.arc3d(i)/sec.L for i in range(n_pts)])
    idx = np.digitize(pos, bins, right=False) - 1
    idx[idx == sec.nseg] = sec.nseg-1
    pts = np.array([[sec.x3d(i),sec.y3d(i),sec.z3d(i)] for i in range(n_pts)])
    centers = np.zeros((sec.nseg,3))
    interp = [False for _ in range(sec.nseg)]
    for i in range(sec.nseg):
        jdx, = np.where(idx == i)
        if len(jdx) > 0:
            centers[i] = pts[jdx,:].mean(axis=0) * 1e-6
        else:
            if i == 0:
                centers[i] = [sec.x3d(0), sec.y3d(0), sec.z3d(0)]
            elif i == sec.nseg-1:
                centers[i] = [sec.x3d(n_pts-1), sec.y3d(n_pts-1), sec.z3d(n_pts-1)]
            else:
                interp[i] = True
    Ve = np.zeros(sec.nseg)
    for i in range(sec.nseg):
        if interp[i]:
            centers[i] = (centers[i-1] + centers[i+1])/2
        Ve[i] = fun(centers[i],E,theta,phi)
    return Ve,centers
