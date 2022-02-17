def onsite():
    return 1.0#


#onsite = 1.0

def onsite_l():
    return 0.0

def onsite_r():
    return 0.0

def hopping():
    return -1.0

def hopping_y():
    return -1.0

def hopping_lx():
    return -1.0

def hopping_ly():
    return -0.10

def hopping_rx():
    return -1.0

def hopping_ry():
    return -0.10

def hopping_lc():
    return -0.5

def hopping_rc():
    return -0.5

def chain_length_x():
    return 1

def chain_length():
    return 1
def chain_length_y():
    return 1

def chain_length_ly():
    return 1

def chemical_potential():
    return 0.0

def temperature():
    return 0.0

def steps():
    return 81


def e_upper_bound():
    return 14.0

def e_lower_bound():
    return -14.0

def hubbard_interaction():
    return 0.3

voltage_r = [-0.05 * i for i in range(41)]

voltage_l = [0.05 * i for i in range(41)]

voltage_step = 0

def energy( ):
    energy = [e_lower_bound()+( e_upper_bound() - e_lower_bound() ) / steps() * x +0.00000000001*1j for x in range( steps() )]
    return energy

