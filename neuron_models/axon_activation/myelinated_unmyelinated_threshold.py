"""
Electric-field activation thresholds for myelinated and unmyelinated axons.

This script constructs simplified myelinated and unmyelinated axon models
using the NEURON simulator. A spatially oriented extracellular electric field
is applied to each axonal segment, and the minimum field amplitude required
to evoke an action potential is estimated across a range of pulse durations.

The resulting strength-duration curves are used to compare the excitability
of myelinated and unmyelinated axon models.
"""

import numpy as np
from neuron import h
import matplotlib.pyplot as plt
import math
# Load NEURON standard run file
h.load_file('stdrun.hoc')

# Setting pulse duration range (in ms) and initial search range for binary threshold search
pulse_durations = np.linspace(0.05, 1, 20)  # From 50 µs to 1 ms
amplitude_min = 1
amplitude_max = 200
  # Reduce simulation time to speed up threshold search
t_start, t_stop, dt = 0, 10, 0.1
# Lists to store threshold values for myelinated and unmyelinated cases
thresholds_myelinated = []
thresholds_unmyelinated = []
th_acc = 0.1e-2   # 0.1%, accuracy of threshold finding
#factor = np.sqrt(np.sqrt(2))  # factor for increasing/decreasing field amplitude during threshold search
factor=1.1
n_binary_step = np.log(3000)/np.log(factor) # maximum number of steps during the search
n_binary_step = (n_binary_step-(n_binary_step%1)+1)

# Axon parameters and stimulus duration range
diam = 2                   # Diameter in microns
node_length = 1            # Length of node in microns
internode_length = diam * 100  # Internode length based on diameter
nseg_per_internode = 10    # Number of segments per internode
num_nodes = 5              # Number of nodes in the myelinated model
nseg_per_cable = 10        # Number of segments in unmyelinated cable
Cm_nonmyelinated = 1       # Capacitance for unmyelinated (uF/cm2)
Cm_myelinated = 0.02       # Capacitance for myelinated (uF/cm2)
Rm_myelinated = 1.125 * 1e3 # Resistance for myelinated (kOhm.cm2)

# Constants for field angle
theta_deg = 180
phi_deg = 0
theta = np.radians(theta_deg)
phi = np.radians(phi_deg)

# Function to get 3D coordinates at a particular arc length
def interpolate_3d_coords(section, arc):
    n3d = int(section.n3d())
    total_arc = section.L
    for i in range(n3d - 1):
        arc_i = section.arc3d(i) / total_arc
        arc_next = section.arc3d(i + 1) / total_arc
        if arc_i <= arc <= arc_next:
            u = (arc - arc_i) / (arc_next - arc_i)
            x = (1 - u) * section.x3d(i) + u * section.x3d(i + 1)
            y = (1 - u) * section.y3d(i) + u * section.y3d(i + 1)
            z = (1 - u) * section.z3d(i) + u * section.z3d(i + 1)
            return x, y, z
    return 0, 0, 0
    
# Function to compute extracellular potential at each segment over time
def compute_extracellular_potential_time(section, time_vec, Ex_vec, Ey_vec, Ez_vec):
    ve_time = []
    for t_idx, t in enumerate(time_vec):
        ve_temp = []
        for i in range(section.nseg):
            arc = (i + 0.5) / section.nseg
            x, y, z = interpolate_3d_coords(section, arc)
            Ex, Ey, Ez = Ex_vec[t_idx], Ey_vec[t_idx], Ez_vec[t_idx]
            ve_seg = -(Ex * x + Ey * y + Ez * z) * 1e-3
            ve_temp.append(ve_seg)
        ve_time.append(ve_temp)
    return np.array(ve_time)
    
def setup_axon(ismyelinated):
    global sections, v_rec, cable, nodes, internodes
    
    sections = []
    
    if ismyelinated == 0:
        # Unmyelinated cable setup
        L = 805
        cable = h.Section(name='cable')
        cable.L = L
        cable.diam = diam
        cable.nseg = nseg_per_cable
        cable.insert('hh')
        cable.insert('extracellular')
        cable.cm = Cm_nonmyelinated
        sections = [cable]
        v_rec = h.Vector().record(cable(0.5)._ref_v)
    else:
        # Myelinated axon setup with nodes and internodes
        nodes, internodes = [], []
        z_pos = 0
        L = num_nodes * node_length + (num_nodes - 1) * internode_length
        
        for i in range(num_nodes):
            node = h.Section(name=f'node_{i+1}')
            node.pt3dclear()
            node.pt3dadd(0, 0, z_pos, diam)
            z_pos += node_length
            node.pt3dadd(0, 0, z_pos, diam)
            node.L = node_length
            node.diam = diam
            node.insert('hh')
            node.insert('extracellular')
            node.nseg = 1
            node.cm = Cm_nonmyelinated
            if 'ca' in node.psection():
                node.uninsert('ca') 
            # Double transient sodium channel conductance for nodes
            for seg in node:
                seg.hh.gnabar *= 2  # Double sodium conductance in node
            nodes.append(node)
        
        for i in range(num_nodes - 1):
            internode = h.Section(name=f'internode_{i+1}')
            internode.pt3dclear()
            internode.pt3dadd(0, 0, z_pos, diam)
            z_pos += internode_length
            internode.pt3dadd(0, 0, z_pos, diam)
            internode.L = internode_length
            internode.diam = diam
            internode.nseg = nseg_per_internode
            internode.insert('hh')
            internode.insert('extracellular')
            internode.cm = Cm_myelinated
            internode.gl_hh = 1 / Rm_myelinated
            internodes.append(internode)
        
        for i in range(num_nodes - 1):
            nodes[i].connect(internodes[i])
            internodes[i].connect(nodes[i + 1])
        
        sections = nodes + internodes
        v_rec = h.Vector().record(internodes[int(num_nodes/2)](0.5)._ref_v)
        
    return sections, v_rec

def find_threshold(duration, ismyelinated):
    pulse_start = 50  # ms
    
    pulse_end = pulse_start + duration
    time_vec = np.arange(t_start, t_stop + dt, dt)
    
    Ex_vec = np.zeros_like(time_vec)
    Ey_vec = np.zeros_like(time_vec)
    Ez_vec = np.zeros_like(time_vec)

    setup_axon(ismyelinated)
    
    low, high = amplitude_min, amplitude_max
    #threshold_amplitude = high

    while (high - low)-1 > 0.1:  # Continue until the range is sufficiently small
        #amplitude = (low + high) / 2
        amp = math.sqrt(high*low)
        for i, t in enumerate(time_vec):
            if pulse_start <= t <= pulse_end:
                Ex_vec[i] = amp * np.sin(theta) * np.cos(phi)
                Ey_vec[i] = amp * np.sin(theta) * np.sin(phi)
                Ez_vec[i] = amp * np.cos(theta)
            else:
                Ex_vec[i] = 0
                Ey_vec[i] = 0
                Ez_vec[i] = 0
        
        ve_cable_times = [
            compute_extracellular_potential_time(section, time_vec, Ex_vec, Ey_vec, Ez_vec)
            for section in sections
        ]
        
        for section, ve_cable_time in zip(sections, ve_cable_times):
            ve_vecs = [h.Vector(ve_cable_time[:, i]) for i in range(section.nseg)]
            for seg_idx, seg in enumerate(section):
                ve_vecs[seg_idx].play(seg._ref_e_extracellular, dt)
        
        t_rec = h.Vector().record(h._ref_t)
        
        h.finitialize(-65)
        h.tstop=t_stop
        h.run()
        
        v = np.array(v_rec)
        spike_occurred = max(v) > 0  # Check if a spike occurred (voltage > 0 mV)
        found_lb=False
        found_ub=False
        if spike_occurred:
            #threshold_amplitude = amp
            high = amp  # Reduce the upper bound
            amp /= factor
            found_ub=True
        else:
            low = amp  # Increase the lower bound
            amp *= factor
            found_lb=True
        if ((low >high) or (high <low)):
                return 0
        amp = math.sqrt(high*low)
        # while not (found_lb and found_ub):
            # if spike_occurred:
                # #threshold_amplitude = amp
                # high = amp  # Reduce the upper bound
                # amp /= factor
                # found_ub=True
            # else:
                # low = amp  # Increase the lower bound
                # amp *= factor
                # found_lb=True
            # if ((low >high) or (high <low)):
                # return 0  
        # amp = math.sqrt(high*low)
    # # return threshold_amplitude
        # found_lb=False
        # found_ub=False
        # while not (found_lb and found_ub):
            # if spike_occurred:  # upper bound is amplitude results in AP
                # high = amp
                # amp /= factor
                # found_ub = True
            # else:
                # low = amp
                # amp *= factor
                
                # found_lb = True
                
            # if ((low >high) or (high <low)):
                # return 0  
        # amp = math.sqrt(high*low)  # use geometric mean of lower and upper bounds
        
        while ((high/low -1) > th_acc):
            if spike_occurred:
                high = amp
                
            else:
                low = amp
                
            amp = math.sqrt(high*low)
            # if stoprun:
                # break
        return high 
# Run the analysis for both myelinated and unmyelinated conditions
for duration in pulse_durations:
    thresholds_unmyelinated.append(find_threshold(duration, ismyelinated=0))
    thresholds_myelinated.append(find_threshold(duration, ismyelinated=1))

# Plot threshold vs. pulse duration
plt.figure(figsize=(10, 6))
plt.plot(pulse_durations * 1e3, thresholds_unmyelinated, label="Unmyelinated", marker='o')
plt.plot(pulse_durations * 1e3, thresholds_myelinated, label="Myelinated", marker='s')
plt.xlabel("Pulse Duration (µs)")
plt.ylabel("Stimulus Threshold (V/m)")
plt.title("Stimulus Threshold vs. Pulse Duration")
plt.legend()
plt.grid(True)
plt.show()
