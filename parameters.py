onsite = 0.0#onsite energy in the scattering region

onsite_l = 0.0  #onsite energy in the left lead

onsite_r = 0.0  #onsite energy in the right lead
    
hopping = -1.0 #the hopping the x direction of the scattering region

hopping_y = -1.0#the hopping the y direction of the scattering region

hopping_lx = -1.0 #the hopping in the x direction of the left lead

hopping_ly = -1.0 #the hopping in the y direction of the left lead

hopping_rx = -1.0 #the hopping in the x direction of the right lead

hopping_ry= -0.10 # the hopping in the y direction of the right lead

hopping_lc = -0.5 # the hopping inbetween the left lead and scattering region

hopping_rc = -0.5 # the hopping inbetween the right lead and scattering region

chain_length = 1 # the number of atoms in the x direction of the scattering region

chain_length_y = 1 # this is the number of k in the y direction for the scattering region

chain_length_ly = 1 # this is the number of k points I will take in the leads in the y direction

chemical_potential = 0.0

temperature = 100

steps = 187 #number of energy points we take

e_upper_bound = 14.0 # this is the max energy value

e_lower_bound = -14.0# this is the min energy value
hubbard_interaction = 0.3 # this is the hubbard interaction

voltage_r = [-0.15 * i for i in range(41)]

voltage_l = [0.15 * i for i in range(41)]

voltage_step = 0 # voltage step of zero is equilibrium. This is an integer and higher values correspond to a higher potential difference between the two leads.

pi = 3.14159265359

if (hubbard_interaction == 0.0):
    interaction_order = 0 # this is the order the green function will be calculated too in terms of interaction strength. this can be equal to 0 , 1 or 2#
else:
    interaction_order = 2
#this needs a tiny imaginary part for convergence in the calculation of the embedding self energy
energy = [e_lower_bound+( e_upper_bound - e_lower_bound ) / steps * x +0.00000000001 * 1j for x in range(steps)]

def conjugate(x):
    a = x.real
    b = x.imag
    y = a - 1j * b
    return y
