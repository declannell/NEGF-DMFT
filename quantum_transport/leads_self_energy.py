import matplotlib.pyplot as plt
import math
import numpy as np
import parameters
import warnings
from typing import List

class EmbeddingSelfEnergy:
    kx: float
    ky: float
    voltage_step : float
    self_energy_left: List[complex]
    self_energy_right: List[complex]
    
    #self_energy_left_lesser: List[complex]
    #self_energy_right_lesser: List[complex]
    """
    transfer_matrix_l: List[complex]
    transfer_matrix_r: List[complex]
    """
    def __init__(self, _kx: float, _ky: float, _voltage_step: int):
        self.voltage_step = _voltage_step
        self.kx = _kx
        self.ky = _ky  
        self.self_energy_left = [0 for r in range(parameters.steps)] 
        self.self_energy_right = [0 for r in range(parameters.steps)]
        #self.self_energy_left_lesser = [0 for r in range(parameters.steps)]
        #self.self_energy_right_lesser = [0 for r in range(parameters.steps)]
        self.get_self_energy()

    def get_self_energy(self):
        #this is based on the paper https://iopscience.iop.org/article/10.1088/0305-4608/14/5/016/meta
        transfer_matrix_l, transfer_matrix_r  = self.get_transfer_matrix()
        surface_gf_l, surface_gf_r = self.sgf(transfer_matrix_l, transfer_matrix_r)
        self.lead_self_energy(surface_gf_l, surface_gf_r)
        #self.lesser_self_energy()
        #self.text_file_retarded()
        #self.text_file_lesser()     
        
    def get_transfer_matrix(self): #this assume t and t_tilde are the same and principal_layer is 1 atom thick
        transfer_matrix_l = [0 for r in range(parameters.steps)] 
        transfer_matrix_r = [0 for r in range(parameters.steps)] 
        t_next_l = [0 for r in range(0, parameters.steps)]
        t_next_r = [0 for r in range(0, parameters.steps)]
        t_product_l = [0 for r in range(0, parameters.steps)]
        t_product_r = [0 for r in range(0, parameters.steps)]        
        
        #print("Left : Epsilon -voltage - 2*hopping*cos(ky) -2*hopping*cos(kx) = ", parameters.onsite_l + parameters.voltage_l[self.voltage_step] + 2 * parameters.hopping_ly * math.cos(self.ky) + 2 * parameters.hopping_lx * math.cos(self.kx))
        #print("Right: Epsilon -voltage - 2*hopping*cos(ky) -2*hopping*cos(kx) = ", parameters.onsite_r + parameters.voltage_r[self.voltage_step] + 2 * parameters.hopping_ry * math.cos(self.ky) + 2 * parameters.hopping_lx * math.cos(self.kx))
  
            #print(-self.parameters.onsite_l-2*self.parameters.hopping_ly*math.cos(k_y))
        for r in range(0, parameters.steps):
            t_next_l[r] = parameters.hopping_lz /(parameters.energy[r] - parameters.onsite_l - parameters.voltage_l[self.voltage_step] - 2 * parameters.hopping_ly * math.cos(self.ky) - 2 * parameters.hopping_lx * math.cos(self.kx))
            t_next_r[r] = parameters.hopping_rz /(parameters.energy[r] - parameters.onsite_r - parameters.voltage_r[self.voltage_step] - 2 * parameters.hopping_ry * math.cos(self.ky) - 2 * parameters.hopping_lx * math.cos(self.kx))
            t_product_l[r] = t_next_l[r]
            t_product_r[r] = t_next_r[r]

            transfer_matrix_l[r] = t_next_l[r]
            transfer_matrix_r[r] = t_next_r[r]   
                

        differencelist = [0 for i in range(0, 2 * parameters.steps)]
        old_transfer = [0 for z in range(0, parameters.steps)] 
        difference = 1       
        count = 0
        while(difference > 0.01):
            count += 1
            for r in range(0, parameters.steps):
                t_next_l[r] = t_next_l[r] ** 2 / (1 - 2 * t_next_l[r] ** 2)
                t_next_r[r] = t_next_r[r] ** 2 / (1 - 2 * t_next_r[r] ** 2)
                t_product_l[r] = t_product_l[r] * t_next_l[r]
                t_product_r[r] = t_product_r[r] * t_next_r[r]
                transfer_matrix_l[r] =transfer_matrix_l[r] + t_product_l[r]
                transfer_matrix_r[r] = transfer_matrix_r[r] + t_product_r[r]
                
            for r in range(0, parameters.steps):                
                differencelist[r] = abs(transfer_matrix_l[r].real - old_transfer[r].real)
                differencelist[parameters.steps + r] = abs(transfer_matrix_l[r].imag - old_transfer[r].imag)
                old_transfer[r] = transfer_matrix_l[r]
            difference = max (differencelist)
            #print ("The difference is ", difference)
        #print(" This converged in " ,count, " iterations.\n")
        return transfer_matrix_l, transfer_matrix_r 
        
    def sgf(self, transfer_matrix_l, transfer_matrix_r):
        surface_gf_l = [0 for r in range(parameters.steps)]
        surface_gf_r = [0 for r in range(parameters.steps)]
        for r in range(0, parameters.steps):
            surface_gf_l[r] = 1 / (parameters.energy[r] - parameters.voltage_l[self.voltage_step] - parameters.onsite_l - 2 * parameters.hopping_ly * math.cos(self.ky) - 2 * parameters.hopping_lx * math.cos(self.kx) - parameters.hopping_lz * transfer_matrix_l[r] )
            surface_gf_r[r] = 1 / (parameters.energy[r] - parameters.voltage_r[self.voltage_step] - parameters.onsite_r - 2 * parameters.hopping_ry * math.cos(self.ky) - 2 * parameters.hopping_lx * math.cos(self.kx) - parameters.hopping_rz * transfer_matrix_r[r] )
        return surface_gf_l, surface_gf_r

    def lead_self_energy(self, surface_gf_l, surface_gf_r):
        for r in range(0,  parameters.steps):
            self.self_energy_left[r] = parameters.hopping_lc ** 2 * surface_gf_l[r]
            self.self_energy_right[r] = parameters.hopping_rc ** 2 * surface_gf_r[r]

    def text_file_retarded(self): #num is the number of k-points
        f = open('/home/declan/green_function_code/green_function/textfiles/embedding_self_energy.txt', 'w')

        for r in range(0, parameters.steps ):
            f.write(str(parameters.energy[r].real) )
            f.write( "," )
            f.write(str(self.self_energy_left[r].real))
            f.write( "," )          
            f.write(str(self.self_energy_left[r].imag))
            f.write( "," )
            f.write(str(self.self_energy_right[r].real))
            f.write( "," )
            f.write(str(self.self_energy_right[r].imag))
            f.write( "," )

        f.close()

    def text_file_lesser(self): #num is the number of k-points
        f = open('/home/declan/green_function_code/green_function/textfiles/embedding_self_energy_lesser.txt', 'w')
        for r in range(0, parameters.steps):
            f.write(str(self.self_energy_left_lesser[r].real))
            f.write( "," )          
            f.write(str(self.self_energy_left_lesser[r].imag))
            f.write( "," )
            f.write(str(self.self_energy_right_lesser[r].real))
            f.write( "," )
            f.write(str(self.self_energy_right_lesser[r].imag))
            f.write( "," )
        f.close()

    def lesser_self_energy(self):
        for r in range(0 , parameters.steps ):#fluctautation dissapation theorem is valid
            self.self_energy_left_lesser[r] = - fermi_function(parameters.energy[r].real - parameters.voltage_l[self.voltage_step]) * (self.self_energy_left[r] - parameters.conjugate(self.self_energy_left[r]))
            self.self_energy_right_lesser[r] = - fermi_function(parameters.energy[r].real - parameters.voltage_r[self.voltage_step]) * (self.self_energy_right[r] - parameters.conjugate(self.self_energy_right[r]))


    def plot_self_energy(self):
        fig = plt.figure()  
        plt.plot([e.real for e in parameters.energy], [e.imag for e in self.self_energy_left], color='blue', label='imaginary self energy') 
        plt.plot([e.real for e in parameters.energy], [e.real for e in self.self_energy_left], color='red', label='real self energy') 
        plt.title(" Numerical left Self Energy")
        plt.legend(loc='upper right')
        plt.xlabel("energy")
        plt.ylabel("Self Energy")  
        plt.show()
        
        fig = plt.figure()
        plt.plot([e.real for e in parameters.energy], [e.imag for e in self.self_energy_right], color='blue', label='imaginary self energy') 
        plt.plot([e.real for e in parameters.energy], [e.real for e in self.self_energy_right], color='red', label='real self energy') 
        plt.title(" Numerical right Self Energy")
        plt.legend(loc='upper right')
        plt.xlabel("energy")
        plt.ylabel("Self Energy")  
        plt.show()

        plt.plot([e.real for e in parameters.energy], [e.imag for e in self.self_energy_left_lesser], color = 'green', label='imaginary')
        plt.plot([e.real for e in parameters.energy], [e.real for e in self.self_energy_left_lesser], color = 'orange', label='real')
        plt.title(" Numerical left Self Energy lesser")
        plt.legend(loc='upper right')
        plt.xlabel("energy")
        plt.ylabel("Self Energy lesser")  
        plt.show()
        
        plt.plot([e.real for e in parameters.energy], [e.imag for e in self.self_energy_right_lesser], color = 'green', label='imaginary')
        plt.plot([e.real for e in parameters.energy], [e.real for e in self.self_energy_right_lesser], color = 'orange', label='real')
        plt.title(" Numerical right Self Energy lesser")
        plt.legend(loc='upper right')
        plt.xlabel("energy")
        plt.ylabel("Self Energy lesser")  
        plt.show()   

def create_matrix(size: int):
    return [[0.0 for x in range(size)] for y in range(size)]

def fermi_function(energy: complex):
    if( parameters.temperature == 0):
        if(energy < parameters.chemical_potential):
            return 1
        else:
            return 0
    else:
        return 1 / (1 + math.exp((energy.real - parameters.chemical_potential) / parameters.temperature))
                
def theta_function(a: float, b:float):
    if( a > b):
        return 1
    else:
        return 0
    
def sgn(x: float):
    if x>0 :
        return 1
    elif x<0 :
        return -1
    
def analytic_se(voltage_step: int):
    analytic_se = [0 for i  in range(parameters.steps)]# this assume the interaction between the scattering region and leads is nearest neighbour 
    for i in range(0 , parameters.steps):
        x = (parameters.energy[i].real - parameters.onsite_l - parameters.voltage_l[voltage_step]) / (2 * parameters.hopping_lz)
        #print(x, energy[i])
        analytic_se[i] = (parameters.hopping_lc ** 2) * (1 / abs(parameters.hopping_lz)) * x 
        if (abs(x) > 1):
            analytic_se[i] = analytic_se[i] - (parameters.hopping_lc ** 2) * (1 / abs(parameters.hopping_lz)) * (sgn(x) * np.sqrt(abs(x) * abs(x) - 1)) 
        elif( abs(x) < 1):
            analytic_se[i] = analytic_se[i] - (parameters.hopping_lc ** 2) * abs((1 / abs(parameters.hopping_lz))) * (1j * np.sqrt(1 - abs(x) * abs(x)))

    #print(analytic_se)
  
    plt.plot( parameters.energy, [e.imag for e in analytic_se], color='blue', label='imaginary self energy' ) 
    plt.plot( parameters.energy, [e.real for e in analytic_se], color='red', label='real self energy') 
    plt.title(" Analytical left Self Energy")
    plt.legend(loc='upper right')
    plt.xlabel("energy")
    plt.ylabel("Self Energy") 
    plt.show()
    

def main():
    kx = [0 for m in range(0, parameters.chain_length_x)]  
    ky = [0 for m in range(0, parameters.chain_length_y)]   
    for i in range(0, parameters.chain_length_y):
        if (parameters.chain_length_y != 1):
            ky[i] = 2 * parameters.pi * i / parameters.chain_length_y
        elif (parameters.chain_length_y == 1):
            ky[i] = parameters.pi / 2.0  
    
    for i in range(0, parameters.chain_length_x):
        if (parameters.chain_length_x != 1):
            kx[i] = 2 * parameters.pi * i / parameters.chain_length_x
        elif (parameters.chain_length_x == 1):
            kx[i] = parameters.pi / 2.0  
 # voltage step of zero is equilibrium.
    print("The voltage difference is ", parameters.voltage_l[parameters.voltage_step] - parameters.voltage_r[parameters.voltage_step])
    
    for i in range(0, parameters.chain_length_x):
        for j in range(0, parameters.chain_length_y):
            self_energy = EmbeddingSelfEnergy( kx[i], ky[j], parameters.voltage_step )
            self_energy.plot_self_energy()
            #analytic_se(parameters.voltage_step)
    #print(self_energy.self_energy_left)

if __name__=="__main__":#this will only run if it is a script and not a import module
    main()