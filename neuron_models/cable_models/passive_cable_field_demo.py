import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from neuron import h
h.load_file('stdrun.hoc')
h.load_file('stdlib.hoc')
if __name__ == '__main__':
    # Parameters
    cm = 1.0  # [uF/cm2] membrane capacitance
    Ra = 100.0  # [Ohm cm] cytoplasmic resistivity
    Rm = 15000.0  # [Ohm cm2] membrane resistance
    El = -65.0  # [mV] passive reversal potential
    diam = 5.0  # [um] (constant) diameter of the cable
    # Compute the membrane time constant
    # taum = Rm * cm * 1e-3  # [ms]
    # print('Membrane time constant: {:.2f} ms.'.format(taum))

    # # Compute lambda, the length constant
    # length_const = np.sqrt((diam * 1e-4 * Rm) / (4 * Ra)) * 1e4
    # print('Length constant: {:.3f} um.'.format(length_const))

    # Choose the length of the cable
    #L = 6 * length_const
    L=800
    # Instantiate the section
    cable = h.Section(name='cable')
    cable.cm = cm
    cable.Ra = Ra
    cable.L = L
    cable.diam = diam
    cable.insert('pas')  # Insert passive channels instead of HH
    cable.g_pas = 1 / Rm
    cable.e_pas = El
    
    #cable.insert('hh')  # Insert Hodgkin-Huxley channels
    cable.insert('extracellular')
    cable.e_extracellular = El  # Set passive reversal potential

    # Set the number of segments
    cable.nseg = int((L / (0.1 * h.lambda_f(100, sec=cable)) + 0.9) / 2) * 2 + 1
    print('The cable is {:g} um long and is subdivided into {} segments.'.format(cable.L, cable.nseg))

    # Electrical stimulation parameters
    
    stim_x = 0.5  # Relative location of stimulation
    stim_strength = 100.0  # Voltage change (mV) for stimulation
    stim_duration = 30  # Duration of the stimulation (ms)
    
    stim = h.IClamp(cable(stim_x))
    stim.delay = 5  # Delay before stimulation
    stim.dur = stim_duration  # Duration of stimulation
    stim.amp = stim_strength * 1e-3  # Current injection in nA (convert mV to nA)

    # # Apply the stimulation to the specific segment
    # for seg in cable:
        # if seg.x >= stim_x - 0.01 and seg.x <= stim_x + 0.01:  # Only stimulate near the target
            # seg.v += stim_strength  # Apply a voltage change

    # Electric field parameters
    E_magnitude = 1  # [mV/um] Electric field strength
    theta_deg = 180  # Polar angle
    phi_deg = 0   # Azimuthal angle
    theta=np.radians(theta_deg)
    phi=np.radians(phi_deg)
    Ex = E_magnitude*np.sin(theta) * np.cos(phi)
    Ey = E_magnitude*np.sin(theta) * np.sin(phi)
    Ez = E_magnitude*np.cos(theta)
    
    sElec=h.Section(name='sElec')
    sElec.insert('extracellular')
    len=800
    sElec.pt3dclear()
    sElec.pt3dadd(0,0,0,1)
    sElec.pt3dadd(len*Ex,len*Ez,-len*Ey,1)   
    # Create a section list and append sElec
    sElec_list = h.SectionList()
    sElec_list.append(sec=sElec)
    for sec in h.allsec():
        for seg in sec:
            sec.pt3dclear()
