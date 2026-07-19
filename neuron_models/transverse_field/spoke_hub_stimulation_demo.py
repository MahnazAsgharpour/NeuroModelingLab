from neuron import h, gui
print("Mah")
import numpy as np
import os
import math
import random 
import matplotlib.pyplot as plt
from math import cos,sin,pi

### This part is from the main_UF_Axon_HH.hoc;  

# Placeholder class for Graph
class Graph:
    def __init__(self):
        pass

    def size(self, xlow, xhigh, ylow, yhigh):
        pass  # Set up the graph size

# Initial setup
R_num = 6
folder_str = "NEURON results"
is_axon = 1
radius_init = 0.25

############################################################################################################################################

###  1. Load NEURON GUI
h.load_file("nrngui.hoc")

#############################################################################################################################################

### 2. This Part is converting the load_file("global_parameter.hoc") to Python  // creates global parameter for the simulation

n_theta = 15

# Simulation parameters
dt_default = 0.002
t_after_pulse = 10  #time after pulse ends, ms
steps_per_ms = 1 
celsius = 23.5
v_init = -65
extra_e_center=0
PW_start = 0.001    #ms
decades = 4         # decades for PW
sample_num_decade = 6
sample_decade = [1, 1.4, 2.1, 3.1, 4.5, 6.8]

# Compartment setup   
if is_axon == 1:
    n_dim = 1           #Dimension parameter, 1 for axon
    length_axon = 10    # Length of axon compartment, 10 um
    compart_str = "Axon"
else:
    n_dim = 2
    compart_str = "Soma"

file_name = f"{folder_str}/{compart_str}_summary.txt"
summary_file = h.File()
summary_file.wopen(file_name)   	#open file for writing
summary_file.printf("Setting initial parameters...     \n")
summary_file.printf ("Creating multi-compartment structure equivelant for %s in transverse field....\n", compart_str)

if (n_theta % 2) == 0:
    n_theta += 1

delta_theta = pi / n_theta
n_TP = n_theta - 1
n_TP_half = n_TP // 2

theta_vec = h.Vector(n_TP)
nRcos_vec = h.Vector(n_TP)
cos_vec = h.Vector(n_TP)
sin_vec = h.Vector(n_TP)

theta_vec.x[0] = delta_theta / 2
for ii in range(1, n_TP):
    theta_vec.x[ii] = theta_vec.x[ii-1] + delta_theta
    if ii == n_TP_half:
        theta_vec.x[ii]=theta_vec.x[ii] + delta_theta
for ii in range(n_TP_half):
    cos_vec.x[ii] = cos(theta_vec.x[ii])
    sin_vec.x[ii] = sin(theta_vec.x[ii])
    cos_vec.x[n_TP-1-ii] = -cos_vec.x[ii]
    sin_vec.x[n_TP-1-ii] = sin_vec.x[ii]

def print_theta():
    print("Theta vector:")
    for theta in theta_vec:
        print(theta * 180 / pi)

PW_vec = h.Vector(sample_num_decade * decades + 1)

for ii in range(decades):
    for jj in range(sample_num_decade):
        PW_vec.x[ii * sample_num_decade + jj] = PW_start * 10**ii * sample_decade[jj]
PW_vec.x[PW_vec.size()-1] = PW_start * 10**decades

# Create a vector for pulse widths (PW)
PW_vec = h.Vector(sample_num_decade * decades + 1)

# Fill PW_vec
for ii in range(decades):
    for jj in range(sample_num_decade):
        PW_vec.x[ii * sample_num_decade + jj] = PW_start * 10**ii * sample_decade[jj]
PW_vec.x[PW_vec.size() - 1] = PW_start * 10**decades  # Last pulse duration

# Function to print PW_vec
def print_PW_vec():
    print("Pulse Width Vector (PW_vec):")
    for value in PW_vec:
        print(value)

th_acc = 0.1e-2   # 0.1%, accuracy of threshold finding
factor = np.sqrt(np.sqrt(2))  # factor for increasing/decreasing field amplitude during threshold search
n_binary_step = np.log(3000) / np.log(factor) # maximum number of steps during the search
n_binary_step = int(n_binary_step - (n_binary_step % 1) + 1)
summary_file.printf("Initial parameters set...\n\n")
print("Initial parameters set...")
print(" ")
#print(f"n_binary_step is {n_binary_step }")
#########################################################################################################################################

### 3. This Part is converting the  load_file("create_cell.hoc") to Python	// specifies topological and biophysical properties of the cell, geometry of each compartment is default

Ra_center=35.4
Ra_TP=1e-3
extra_e_center = 0
# Initialize section lists
all_sec = h.SectionList()
TP_sec = h.SectionList()

# Hub compartment
center = h.Section(name='center')  
#center.insert('cellm')  # Inserting the cellm mechanism
center.nseg = 1
center.cm = 1
#center.Ra = Ra_center            # Axial resistance for the center section
center.insert('hhh')
center.insert('extracellular')           # Inserting the extracellular mechanism
center.e_extracellular = extra_e_center
center.Ra = Ra_center            # Axial resistance for the center section


all_sec.append(sec=center)

# Spoke compartments

TPcompart = [h.Section(name=f'TPcompart[{ii}]') for ii in range(n_TP)]
for ii, sec in enumerate(TPcompart):
    sec.connect(center(0.5))  # Connect to node of hub
    all_sec.append(sec=sec)
    TP_sec.append(sec=sec)


for sec in TP_sec:
    sec.nseg = 1
    sec.cm = 1
    sec.insert('hhh')  # Insert HH channels
    sec.insert('extracellular')
    sec.e_extracellular = extra_e_center
    sec.Ra = Ra_TP

for sec in h.allsec():
    print(f"Section {sec.name()}:")
    sec_info = sec.psection()
    mechanisms = sec_info.get('mechanism', [])
    for mech in mechanisms:
        print(f"  Mechanism: {mech}")
        #if mech == 'extracellular':
        if mech == 'hhh':
            #print(f"    extracellular mechanism is present in {sec.name()}")
            print(f"    hhh mechanism is present in {sec.name()}")
# Summary
summary_file.printf("Multi-compartment structure created.\n\n")
print("Multi-compartment structure created. Topology as follows:")
h.topology()
print(" ")

###################################################################################################################################

### 4. This Part is converting the  load_file("set_morph_func.hoc") to Python // procedure to change morphology of the cell for TP

# Define the set_morph function

def set_morph(radius, length):
    global cos_vec, sin_vec,TPcompart,center,n_TP,n_theta,Ra_center
    
    total_area = 2 * pi * radius * length  # Surface area of the single compartment cell
    #nRcos_vec_temp = h.Vector(n_TP)
    if is_axon == 1:
        TP_area = total_area / n_theta  # Area of each compartment in the spoke-hub structure
        TP_radius = np.sqrt(TP_area / (4 * pi)) # For simplicity it was assumed that the Length of spoke compartments is equal to their diameter

        center.pt3dclear()
        center.pt3dadd(0, 0, -TP_radius, TP_radius * 2)
        center.pt3dadd(0, 0, TP_radius, TP_radius * 2)
        center.Ra= Ra_center * (length / 2) * (TP_radius / (radius**2)) #
                    
        for ii in range(int(n_TP)):
            nRcos_vec.x[ii] = (1 + 1 / n_dim) * radius * cos_vec.x[ii]  # um
            TPcompart[ii].pt3dclear()
            TPcompart[ii].pt3dadd(0, 0, 0, TP_radius * 2)
            TPcompart[ii].pt3dadd(TP_radius * 2 * cos_vec.x[ii], TP_radius * 2 * sin_vec.x[ii], 0, TP_radius * 2)

    else:  # For soma
        TP_area = total_area * 1 * pi / (2 * n_theta)  # The 1 here is for sin(pi/2)
        TP_radius = np.sqrt(TP_area / (4 * pi))
        center.pt3dclear()
        center.pt3dadd(0, 0, -TP_radius, TP_radius * 2)
        center.pt3dadd(0, 0, TP_radius, TP_radius * 2)
        center.Ra = Ra_TP

        for ii in range(int(n_TP)):
            nRcos_vec.x[ii] = (1 + 1 / n_dim) * radius * cos_vec.x[ii]
            TP_area = total_area * sin_vec.x[ii] * pi / (2 * n_theta)
            TP_radius = np.sqrt(TP_area / (4 * pi))
            TPcompart[ii].pt3dclear()
            TPcompart[ii].pt3dadd(0, 0, 0, TP_radius * 2)
            TPcompart[ii].pt3dadd(TP_radius * 2 * cos_vec.x[ii], TP_radius * 2 * sin_vec.x[ii], 0, TP_radius * 2)
            
    return center, TPcompart,nRcos_vec 

    
############################################################################################################################################    
    
### 5. This Part is converting the load_file("set_time_vec.hoc") to Python // procedure to set time vector and stimulus vector

# Defining the function
def set_stim_vector(PW, dt_set):
    
    n_time_before = 5   # time steps before pulse onset (t = 0)
    n_time = int(tstop / dt_set)  # time steps from pulse onset to stop time
    n_time = n_time - (n_time % 1) + n_time_before + 1  # time steps rounded, and with steps before pulse onset
    PW_residual = PW % dt_set
    
    if PW_residual != 0:
        n_time += 1  # If pulse duration is not multiple of time step, then include the pulse duration in the vector
    
    time_vec = h.Vector(n_time)
    stim_vec_temp = h.Vector(time_vec.size())
    stim_vec_temp.fill(0)
    
    for ii in range(n_time_before-1, -1, -1):  # negative time
        time_vec.x[ii] = time_vec.x[ii + 1] - dt_set
    
    ii = n_time_before
    while time_vec.x[ii - 1] + dt_set < PW:
        time_vec.x[ii] = time_vec.x[ii - 1] + dt_set
        stim_vec_temp.x[ii] = 1  # Stimulation on
        ii += 1
    
    time_vec.x[ii] = PW  # Pulse duration
    stim_vec_temp.x[ii] = 1
    
    if PW_residual != 0:  # Next time step should be on a multiple
        ii += 1
        #time_vec.x[ii] = time_vec.x[ii - 1] + (dt_set - PW_residual)
        time_vec.x[ii] =PW
    for jj in range(ii + 1, time_vec.size()):
        time_vec.x[jj] = time_vec.x[jj - 1] + dt_set
    return time_vec, stim_vec_temp

#############################################################################################################################################

### 6. This Part is converting the load_file("stimulate.hoc") to Python // function: runs simulation, determines whether cell fires

# Function to handle the stimulation and recording process
def stimulate(amplitude):  # amplitude of stimulus
    global stim_vec_temp, time_vec, nRcos_vec_temp, log_file, v_init,TPcompart,center,v_plot, h
    stim_vec = h.Vector()
    v_plot = h.Graph()
    v_plot.size(0, time_vec.x[-1], -75, 40)
    #v_plot=h.Graph()
    # Clone stim_vec_temp and scale it by the amplitude
    stim_vec.copy(stim_vec_temp)
    stim_vec.mul(amplitude)
    
    # Initialize vectors for voltage recording
    v_vec = h.Vector(time_vec.size(), v_init)
    v_vec_pos_TP = h.Vector(time_vec.size(), v_init)
    v_vec_neg_TP = h.Vector(time_vec.size(), v_init)
    
    flag_AP = 0
    
    # Initialize the simulation
    h.stdinit()
    h.finitialize(v_init)
    v_vec.x[0] = v_init
    v_vec_pos_TP.x[0] = v_init
    v_vec_neg_TP.x[0] = v_init
    
    
    # Run the simulation
    for ii in range(1, int(time_vec.size())):
    #for ii in range(1,time_vec.size()):
        if h.dt != (time_vec.x[ii] - time_vec.x[ii-1]):
            h.dt = time_vec.x[ii] - time_vec.x[ii-1]
        
        # Update extracellular potential for each TP compartment
        for jj in range(n_TP):
            TPcompart[jj].e_extracellular = (extra_e_center - nRcos_vec.x[jj] * stim_vec.x[ii])
        h.fadvance()
        
        # Record the voltages
        v_vec.x[ii] = center(0.5).v
        v_vec_pos_TP.x[ii] = TPcompart[0](0.5).v
        v_vec_neg_TP.x[ii] = TPcompart[n_TP-1](0.5).v
        
        print(f"Time step {ii}, Voltage at center(0.5): {center(0.5).v}")
        if center(0.5).v > 0:
            flag_AP = 1
    ## Plot the results
    if flag_AP:
        v_vec.line(v_plot, time_vec, 1, 5)
        v_vec_pos_TP.line(v_plot, time_vec, 2, 5)
        v_vec_neg_TP.line(v_plot, time_vec, 3, 5)
    else:
        v_vec.line(v_plot, time_vec, 1, 1)
        v_vec_pos_TP.line(v_plot, time_vec, 2, 1)
        v_vec_neg_TP.line(v_plot, time_vec, 3, 1)
    h.cvode.event(tstop, h.quit)  # Schedule the NEURON simulation to quit when done    
    return flag_AP, v_vec

dt_set=0.002
PW=0.01
tstop = PW + t_after_pulse
time_vec, stim_vec_temp=set_stim_vector(PW, dt_set)
length_axon=10
amplitude=4666
radius=radius_init
center, TPcompart, nRcos_vec=set_morph(radius,length_axon)
print(f"Morphology set with new radius of {radius} um.")
v_plot = h.Graph()
# v_plot.size(0, PW + 5, -75, 40)
#v_plot.size(0, PW + 20, -100, 60)
flag_AP,v_vec=stimulate(amplitude)

def print_v_vec():
    print("Voltage vector (v_vec):")
    for value in v_vec:
        print(value)

def print_time_vec():
    print("Time vector (time_vec):")
    for value in time_vec:
        print(value)

V=center(0.5).v
print(f"center voltage is {V}.")
fig, (ax1, ax2) = plt.subplots(2)
fig.suptitle("Nazi Plot")
ax1.plot(time_vec, stim_vec_temp, drawstyle='steps-post')
ax1.set_title('Stimulation Vector Over Time')
ax1.set_xlabel('Time (ms)')
ax1.set_ylabel('Stimulation')

ax2.plot(time_vec, v_vec, drawstyle='steps-post')
ax2.set_xlabel('Time (ms)')
ax2.set_ylabel('Membrane Potential(mV)')
ax2.set_title('Membrane Voltage of the Center')


fig.savefig("123.png")

if flag_AP==1:
    print("AP")
else: 
    print("No AP")
#h.continuerun(tstop)  # Start the NEURON simulation and keep the GUI open
#print_PW_vec()
