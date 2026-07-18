"""
Transverse TMS electric-field threshold simulation.

This script creates a spoke–hub multicompartment morphology in NEURON
to approximate transverse polarization across an axonal or somatic
compartment. An extracellular electric field is applied to the spoke
compartments, and the minimum field amplitude required to evoke an
action potential is estimated for different geometries and pulse widths.

Main components:
1. Construction of the stimulation time vector.
2. Creation of the spoke–hub transverse morphology.
3. Application of extracellular electric potentials.
4. Detection of action-potential generation.
5. Iterative electric-field threshold estimation.
"""
from neuron import h
import numpy as np
import os
import math
import random 
import matplotlib.pyplot as plt
from math import cos,sin,pi
    
############################################################################################################################################    
    
### This Part is converting the load_file("set_time_vec.hoc") to Python // procedure to set time vector and stimulus vector

def set_stim_vector(PW, dt_set):
    
    n_time_before = 5   # time steps before pulse onset (t = 0)
    n_time =(tstop/dt_set)  # time steps from pulse onset to stop time
    n_time =n_time-(n_time%1)+n_time_before+1  # time steps rounded, and with steps before pulse onset
    #n_time =n_time-(n_time-math.floor(n_time/1))+n_time_before+1
    PW_quotient=math.floor(PW/dt_set)
    estimated_value=PW_quotient*dt_set
    PW_residual=PW - estimated_value
    
    if PW_residual !=0:
        n_time +=1  # If pulse duration is not multiple of time step, then include the pulse duration in the vector
    
    time_vec = h.Vector(n_time)
    stim_vec_temp = h.Vector(time_vec.size(),0)
   
    for ii in range(n_time_before-1, -1, -1):  # negative time
        time_vec.x[ii] = time_vec.x[ii+1]-dt_set
    
    ii = n_time_before
    while ((time_vec.x[ii - 1] + dt_set) < PW):
        ii += 1
        time_vec.x[ii] = time_vec.x[ii-1] + dt_set
        stim_vec_temp.x[ii] = 1  # Stimulation on

    time_vec.x[ii] = PW     # Pulse duration
    stim_vec_temp.x[ii] = 1
    if PW_residual !=0:  # Next time step should be on a multiple
        ii += 1
        time_vec.x[ii] = time_vec.x[ii-1] + (dt_set-PW_residual)
    
    for jj in range(ii+1, time_vec.size()):
        time_vec.x[jj] = time_vec.x[jj-1] + dt_set    
    return time_vec, stim_vec_temp

#############################################################################################################################################

###  This Part is converting the load_file("stimulate.hoc") to Python // function: runs simulation, determines whether cell fires

def stimulate(amplitude):  # amplitude of stimulus
    stim_vec = h.Vector()
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
    
    for ii in range(1,time_vec.size()):
        if h.dt != (time_vec.x[ii] - time_vec.x[ii-1]):
            h.dt = time_vec.x[ii] - time_vec.x[ii-1]
        
        for center_idx in range(n_center):  # Loop over each center
            for jj in range(int(n_TP)):  # Loop over compartments within that center
                TPcompart[center_idx][jj].e_extracellular = (extra_e_center - (nRcos_vec.x[jj] * stim_vec.x[ii]))
        h.fadvance()
        
        
        # Record the voltages
        v_vec.x[ii] = centers[0](0.5).v
        v_vec_pos_TP.x[ii] = TPcompart[0][0](0.5).v
        v_vec_neg_TP.x[ii] = TPcompart[0][n_TP-1](0.5).v
        
        if centers[0](0.5).v > 0:
            flag_AP = 1
            
    v_array=np.array(v_vec)   
    v_array_pos_TP=np.array(v_vec_pos_TP)
    v_array_neg_TP=np.array(v_vec_neg_TP)
    time_array = np.array(time_vec)
    
    plt.figure(figsize=(10, 6))
    
    if flag_AP:
        plt.plot(time_array, v_array, label='Center', color='b', linewidth=2)
        plt.plot(time_array, v_array_pos_TP, label='Positive TP', color='g', linewidth=2)
        plt.plot(time_array, v_array_neg_TP, label='Negative TP', color='r', linewidth=2)
    else:
        plt.plot(time_array, v_array, label='Center', color='b', linestyle='--')
        plt.plot(time_array, v_array_pos_TP, label='Positive TP', color='g', linestyle='--')
        plt.plot(time_array, v_array_neg_TP, label='Negative TP', color='r', linestyle='--')   
    #plt.show()
    return flag_AP
    
############################################################################################################################################

### 7. This Part is converting the load_file("find_thresh.hoc") to Python // threshold search, calls stimulate

def find_threshold(amp, low, high):
    if amp == 0:
        amp = 1
    log_file.printf("Stimulation with amplitude %2.4f\n", amp)
    print("        Stimulation with amplitude", amp)
    found_lb = False
    found_ub = False
    
    while not (found_lb and found_ub):
        if stimulate(amp):  # upper bound is amplitude results in AP
            high = amp
            amp /= factor
            log_file.printf("AP. Decreasing amplitude to %2.4f\n", amp)    
            found_ub = True
            print("AP. Decreasing amplitude to", amp)    
            
        else:
            low = amp
            amp *= factor
            log_file.printf("No AP. Increasing amplitude to %2.4f\n", amp)
            found_lb = True
            print("        No AP. Increasing amplitude to", amp)
        
        if ((low >high) or (high <low)):
            return 0

    print(" ")
    amp = math.sqrt(high*low)  # use geometric mean of lower and upper bounds
    
    while ((high/low -1) > th_acc):
        log_file.printf("Testing new amplitude: %2.5f.   ", amp)
        print("        Testing new amplitude ", amp)
        if stimulate(amp):
            high = amp
            log_file.printf("AP. New lower and upper bounds: %2.5f   %2.5f. \n", low, high)
            print("        AP. Decreasing upper bound ", high)
        else:
            low = amp
            log_file.printf("No AP. New lower and upper bounds: %2.5f   %2.5f. \n", low, high)
            print("        No AP. Increasing lower bound ", low)
        amp = math.sqrt(high*low)
        plt.show()
        if stoprun:
            #plt.show()
            break
    return high 
    
#############################################################################################################################################

if __name__ == "__main__":
    
    class Graph:
        def __init__(self):
            pass

        def size(self, xlow, xhigh, ylow, yhigh):
            pass  

    # Initial setup
    
    R_num = 1
    folder_str = "NEURON results"
    is_axon = 1
    radius_init = 0.25

    ############################################################################################################################################

    ###  1. Load NEURON GUI
    h.load_file("nrngui.hoc")
    #############################################################################################################################################

    ### 2. This Part is converting the load_file("global_parameter.hoc") to Python  

    n_theta = 5

    # Simulation parameters
    dt_default = 0.002
    t_after_pulse = 10  #time after pulse ends, ms
    steps_per_ms = 1 
    celsius = 23.5
    v_init = -65
    extra_e_center=0
    PW_start = 0.001    #ms
    decades = 0         # decades for PW
    sample_num_decade = 6
    sample_decade = [1, 1.4, 2.1, 3.1, 4.5, 6.8]

    # Compartment setup   
    if is_axon == 1:
        n_dim = 1           #Dimension parameter, 1 for axon
        length_axon = 50    # Length of axon compartment, 10 um
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

    ## Create a vector for pulse widths (PW)
    
    PW_vec = h.Vector(sample_num_decade * decades + 1)
    for ii in range(decades):
        for jj in range(sample_num_decade):
            PW_vec.x[ii * sample_num_decade + jj] = PW_start * 10**ii * sample_decade[jj]
    PW_vec.x[PW_vec.size() - 1] = PW_start * 10**decades        # Last pulse duration

    th_acc = 0.1e-2   # 0.1%, accuracy of threshold finding
    factor = np.sqrt(np.sqrt(2))  # factor for increasing/decreasing field amplitude during threshold search
    n_binary_step = np.log(3000)/np.log(factor) # maximum number of steps during the search
    n_binary_step = (n_binary_step-(n_binary_step%1)+1)
    summary_file.printf("Initial parameters set...\n\n")
    print("Initial parameters set...")
    print(" ")
    print(f"n_binary_step is {n_binary_step }")
    #########################################################################################################################################
    ### 3. This Part is converting the  load_file("create_cell.hoc") to Python	
    
    Ra_center=35.4
    Ra_TP=1e-3
    extra_e_center = 0
    n_center=5
    # Initialize section lists
    all_sec = h.SectionList()
    TP_sec = h.SectionList()

    # Hub compartments (centers)
    centers = [h.Section(name=f'center[{ii}]') for ii in range(n_center)]
    
    # Appending center sections to the section list
    for center in centers:
        #center.Ra = Ra_center  # Axial resistance for the center section
        center.insert('extracellular')
        center.e_extracellular = extra_e_center
        all_sec.append(sec=center)
    
    # Spoke compartments
    TPcompart=[]
    
    # Creating TP compartments and connecting them to each center
    
    for i_center in range(n_center):
        tp_sections = [h.Section(name=f'TPcompart[{i_center}][{ii}]') for ii in range(n_TP)]
        TPcompart.append(tp_sections)

    # Connecting TP compartments to the current center
        for sec in tp_sections:
            sec.connect(centers[i_center](0.5))  # Connect to the node of the corresponding center
            sec.Ra = Ra_TP                       # Axial resistance for TP compartment
            all_sec.append(sec=sec)
            TP_sec.append(sec=sec)

# Insert additional properties for all sections
    for sec in all_sec:
        sec.nseg = 1
        sec.cm = 1
        sec.insert('extracellular')
        sec.e_extracellular = extra_e_center
        sec.insert('hhh')  
        
    # Summary
    summary_file.printf("Multi-compartment structure created.\n\n")
    print("Multi-compartment structure created. Topology as follows:")
    h.topology()
    print(" ")

    # Define required objects and variables
    g = None
    log_file = None
    PW_str = ""
    log_ID_str = ""
    r = random.Random()
    
    # Main loop
stoprun = True

for ii in range(R_num):
    
    radius = radius_init * (2**ii)
    center_length = length_axon/n_center
    total_area = 2 * pi * radius * center_length # Surface area of the single compartment cell
    TP_area = total_area / n_theta         # Area of each compartment in the spoke-hub structure
    TP_radius = np.sqrt(TP_area / (4 * pi))  # Assumes the length of spoke compartments equals their diameter

    for center_idx in range(n_center):
        # Update the geometry of each center
        centers[center_idx].pt3dclear()
        centers[center_idx].pt3dadd(0, 0, -TP_radius, TP_radius * 2)
        centers[center_idx].pt3dadd(0, 0, TP_radius, TP_radius * 2)
        centers[center_idx].Ra = Ra_center * (center_length / 2) * (TP_radius / (radius**2))
        
        # Update the corresponding TP compartments for this center
        for jj in range(int(n_TP)):
            nRcos_vec.x[jj] = (1 + (1 / n_dim)) * radius * cos_vec.x[jj]  # um
            TPcompart[center_idx][jj].pt3dclear()
            TPcompart[center_idx][jj].pt3dadd(0, 0, 0, TP_radius * 2)
            TPcompart[center_idx][jj].pt3dadd(TP_radius * 2 * cos_vec.x[jj], TP_radius * 2 * sin_vec.x[jj], 0, TP_radius * 2)
        
        # Print/logging updates
        print(f"Morphology set for center {center_idx} with new radius of {radius} um.")
        summary_file.printf("\nRadius of %s is %1.2f um.\n", compart_str, radius)
        summary_file.printf("Morphology set for center %d with new radius of %1.1f um.\n", center_idx, radius)       
        
        for kk in range(len(PW_vec)):
            parameter_ID = (ii * len(PW_vec) + kk + 1)
            PW = PW_vec.x[kk]
            tstop = PW + t_after_pulse
            
            # Create the log file path
            log_ID_str = f"{folder_str}/{compart_str}/{parameter_ID}.txt"
            log_file = h.File()
            log_file.wopen(log_ID_str)
            plt.show()
            #v_plot = h.Graph()
            if PW < 0.05:
                PW_str = f"{PW * 1000:.1f} us"
                dt_set = PW / 5
            else:
                dt_set = dt_default
                if PW < 1:
                    PW_str = f"{PW * 1000:.0f} us"
                else:
                    PW_str = f"{PW:.1f} ms"
                    
            time_vec, stim_vec_temp = set_stim_vector(PW, dt_set)
            
            print("    Pulse width is ", PW_str, ". Parameter ID is ", parameter_ID)
            log_file.printf(f"Compartment type: {compart_str}\nParameter ID: {parameter_ID}\nRadius: {radius:.2f} um\nPulse duration: {PW_str}\n")
            log_file.printf("\n-----------------------------------------------------------\n")
            log_file.printf("Search for threshold...\n")
            
            if PW < 1:
                E_init = 1/PW/radius * (1+0*r.normalvariate(1,0.005))
            else:
                E_init = 1/radius * (1+0*r.normalvariate(1,0.005))
                
            lb = E_init/(factor**n_binary_step)
            ub = E_init*(factor**n_binary_step)
            E_threshold = find_threshold(E_init, lb, ub)
            #plt.show()
            #plt.clf()
            log_file.printf("\n-----------------------------------------------------------\n")
            log_file.printf(f"Threshold search finished: \n\tE-field: {E_threshold * 10**3:.5f} mV/mm\n\tE*R: {E_threshold * radius:.5f} mV\n")
            log_file.close()
            #plt.close()
            summary_file.printf(f"Pulse width is {PW_str}. Parameter ID is {parameter_ID}. Threshold is {E_threshold * 10**3:.6f} mV/mm.\n")
            print("    Threshold search finished ", E_threshold * 10**3, " mV/mm ", E_threshold * radius, " mV")
            print("")
        print("")
    summary_file.close()
