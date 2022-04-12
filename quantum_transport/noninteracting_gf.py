import scipy.linalg as la
import matplotlib.pyplot as plt
import math
import time
import leads_self_energy
import parameters
import warnings
from typing import List

class Noninteracting_GF:
    kx: float
    ky: float
    voltage_step: float
    hamiltonian: List[List[complex]]   
    effective_hamiltonian: List[List[List[complex]]]  
    noninteracting_gf: List[List[List[complex]]]  

    def __init__(self, _kx: float, _ky: float, _voltage_step: int):
        self.kx = _kx
        self.ky = _ky
        self.voltage_step = _voltage_step
        self.hamiltonian = create_matrix(parameters.chain_length)
        self.effective_hamiltonian = [create_matrix(parameters.chain_length) for r in range(parameters.steps)]
        self.noninteracting_gf = [create_matrix(parameters.chain_length) for r in range(parameters.steps)]       
        #this willgetting the embedding self energies from the leads code        
        self.get_effective_matrix()
        self.get_noninteracting_gf()

    def get_effective_matrix(self):
        self_energy = leads_self_energy.EmbeddingSelfEnergy(self.kx, self.ky, parameters.voltage_step)
        #self_energy.plot_self_energy()
        for i in range(0, parameters.chain_length-1):
            self.hamiltonian[i][i+1] = parameters.hopping 
            self.hamiltonian[i+1][i] = parameters.hopping
        for i in range(0, parameters.chain_length):
            voltage_i = parameters.voltage_l[self.voltage_step] - (i + 1) / (float)(parameters.chain_length + 1) * (parameters.voltage_l[self.voltage_step] - parameters.voltage_r[self.voltage_step])
            print("The external voltage is on site ",  i , " is ", voltage_i)
            self.hamiltonian[i][i] = parameters.onsite + 2 * parameters.hopping_x * math.cos(self.kx) + 2 * parameters.hopping_y * math.cos(self.ky) + voltage_i
            for j in range(0,parameters.chain_length):  
                for r in range(0, parameters.steps):
                    self.effective_hamiltonian[r][i][j] = self.hamiltonian[i][j]
        
        for r in range(0, parameters.steps):
            self.effective_hamiltonian[r][0][0] += self_energy.self_energy_left[r]
            self.effective_hamiltonian[r][-1][-1] += self_energy.self_energy_right[r]
        """    
        plt.plot(parameters.energy, [e[0][0].real for e in self.effective_hamiltonian], color='red', label='real effective hamiltonian') 
        plt.plot(parameters.energy, [e[0][0].imag for e in self.effective_hamiltonian], color='blue', label='Imaginary effective hamiltonian')
        plt.title("effective hamiltonian")
        plt.legend(loc='upper left')
        plt.xlabel("energy")
        plt.ylabel("effective hamiltonian")  
        plt.show()   
        """
    def get_noninteracting_gf(self):
        inverse_green_function = create_matrix(parameters.chain_length)
        for r in range(0, parameters.steps):
            for i in range(0, parameters.chain_length):
                for j in range(0, parameters.chain_length):
                    if (i == j):
                        inverse_green_function[i][j] = parameters.energy[r].real - self.effective_hamiltonian[r][i][j]
                    else:
                        inverse_green_function[i][j] = - self.effective_hamiltonian[r][i][j]

            self.noninteracting_gf[r] = la.inv(inverse_green_function, overwrite_a=False, check_finite=True)

    def plot_greenfunction(self):
        for i in range(0, parameters.chain_length):

            plt.plot(parameters.energy, [e[i][i].real for e in self.noninteracting_gf], color='red', label='Real Green up') 
            plt.plot(parameters.energy, [e[i][i].imag for e in self.noninteracting_gf], color='blue', label='Imaginary Green function')
            j = i +1
            plt.title('Noninteracting Green function site %i' %j)
            plt.legend(loc='upper left')
            plt.xlabel("energy")
            plt.ylabel("Noninteracting green Function")  
            plt.show()     

    def print_hamiltonian(self):# this allows me to print the effective hamiltonian if called for a certain energy point specified by num. 
    # eg. hamiltonian.print(4) will print the effective hamiltonian of the 4th energy step
        for i in range(0,parameters.chain_length):
            row_string = " ".join((str(r).rjust(5, " ") for r in self.hamiltonian[i])) #rjust adds padding, join connects them all
            print(row_string)

def analytic_gf_1site(gf_int_up):#this the analytic soltuion for the noninteracting green function when we have a single site in the scattering region
    analytic_gf = [ 0 for i  in range( parameters.steps ) ]# this assume the interaction between the scattering region and leads is nearest neighbour 
 
    self_energy = leads_self_energy.EmbeddingSelfEnergy(self.kx, self.ky, parameters.voltage_step)

    for r in range( 0 , parameters.steps ):   
        x = parameters.energy[r].real - parameters.onsite - self_energy.self_energy_left[r].real - self_energy.self_energy_right[r].real
        y = self_energy.self_energy_left[r].imag + self_energy.self_energy_right[r].imag
        analytic_gf[r] = x / ( x * x + y * y ) + 1j * y / ( x * x +y * y )
  

    plt.plot(parameters.energy , [ e[0][0].real for e in gf_int_up] , color='red' , label='real green function' )
    plt.plot(parameters.energy , [ e[0][0].imag for e in gf_int_up], color='blue', label='imaginary green function' )
    plt.plot( parameters.energy , [ e.imag for e in analytic_gf ], color='blue', label='analytic imaginary green function' ) 
    plt.plot( parameters.energy , [e.real for e in analytic_gf] , color='red' , label='analytic real green function') 
    plt.title(" Analytical Green function and numerical GF")
    #plt.legend(loc='upper right')
    plt.xlabel("energy")
    plt.ylabel("Green Function")  
    plt.show()

def create_matrix(size: int):
    return [ [ 0.0 for x in range( size ) ] for y in range( size )]

def analytic_gf_1site(gf_int_up: List[List[List[float]]], kx: float, ky: float):#this the analytic soltuion for the noninteracting green function when we have a single site in the scattering region
    analytic_gf = [ 0 for i  in range( parameters.steps ) ]# this assume the interaction between the scattering region and leads is nearest neighbour 
    energy = [ 0 for i in range( parameters.steps ) ]   
    
    self_energy = leads_self_energy.EmbeddingSelfEnergy(parameters.pi/2.0, parameters.pi/2.0, parameters.voltage_step) 
    f = open('/home/declan/green_function_code/green_function/textfiles/embedding_self_energy.txt', 'r')
    lines = f.read().split(',')  
    for r in range( 0 , parameters.steps ):  
        energy[r] = float( lines[ 5 * r ] )   
        x = parameters.energy[r].real - parameters.onsite - 2 * parameters.hopping_x * math.cos(kx) - 2 * parameters.hopping_y * math.cos(ky) - self_energy.self_energy_right[r].real - self_energy.self_energy_left[r].real
        y = self_energy.self_energy_right[r].imag + self_energy.self_energy_left[r].imag
        analytic_gf[r] = x / ( x * x + y * y ) + 1j * y / ( x * x +y * y )
    f.close()

    plt.plot(parameters.energy , [ e[0][0].real for e in gf_int_up] , color='red' , label='real green function' )
    plt.plot(parameters.energy , [ e[0][0].imag for e in gf_int_up], color='blue', label='imaginary green function' )
    plt.title(" Analytical Green function and numerical GF, single site")
    #plt.legend(loc='upper right')
    plt.xlabel("energy")
    plt.ylabel("Green Function")  
    plt.show()

def analytic_gf_2site(gf_int_up: List[List[List[float]]], kx: float, ky: float):#this the analytic soltuion for the noninteracting green function when we have 2 sites in the scattering region
    analytic_gf= [ 0 for i  in range( parameters.steps ) ]# this assume the interaction between the scattering region and leads is nearest neighbour 
    energy = [ 0 for i in range( parameters.steps ) ]   
    self_energy = leads_self_energy.EmbeddingSelfEnergy(parameters.pi/2.0, parameters.pi/2.0, parameters.voltage_step) 

    f = open('/home/declan/green_function_code/green_function/textfiles/embedding_self_energy.txt', 'r')
    lines = f.read().split(',')  
    print("- 2 * parameters.hopping_x * math.cos(kx) - 2 * parameters.hopping_y * math.cos(ky) = ", - 2 * parameters.hopping_x * math.cos(kx) - 2 * parameters.hopping_y * math.cos(ky))
    for r in range( 0 , parameters.steps ):    
        x = (parameters.energy[r].real) - parameters.onsite  - self_energy.self_energy_right[r].real
        y = - self_energy.self_energy_right[r].imag
        a = x * x - y * y - parameters.hopping * parameters.hopping 
        b = 2 * x * y
        analytic_gf[r] =  ( a * x + b * y ) / ( a * a + b * b ) + 1j * ( y * a - x * b ) / ( a * a + b * b )
    
    f.close()
    #plt.plot(parameters.energy , [ e[0][0].real for e in gf_int_up] , color='red' , label='real green function' )
    #plt.plot(parameters.energy , [ e[0][0].imag for e in gf_int_up], color='blue', label='imaginary green function' )
    plt.plot( parameters.energy , [ e.imag for e in analytic_gf ], color='blue', label='analytic imaginary green function' ) 
    plt.plot( parameters.energy , [e.real for e in analytic_gf] , color='red' , label='analytic real green function') 
    plt.title(" Analytical Green function and numerical GF for two sites")
    #plt.legend(loc='upper right')
    plt.xlabel("energy")
    plt.ylabel("Green Function")  
    plt.show()

def get_spin_occupation( gf_lesser_up: List[complex] , gf_lesser_down: List[complex]  ):#this should work as in first order interaction, it gives the same result as fluctuation dissaption thm to 11 deciaml places
    delta_energy = (parameters.e_upper_bound - parameters.e_lower_bound )/parameters.steps
    result_up , result_down = 0 , 0 
    for r in range( 0 , parameters.steps ):
        result_up = (delta_energy) * gf_lesser_up[r] + result_up
        result_down = (delta_energy) * gf_lesser_down[r] + result_down
    x= -1j / (2 * parameters.pi) * result_up 
    y= -1j / (2 * parameters.pi) * result_down     
    return x , y


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
            noninteracting_gf = Noninteracting_GF(kx[i], ky[j], parameters.voltage_step)
            noninteracting_gf.print_hamiltonian()
            noninteracting_gf.plot_greenfunction()

        if (parameters.chain_length == 1):
            analytic_gf_1site(noninteracting_gf.noninteracting_gf, kx[j], ky[i])
        if (parameters.chain_length == 2 and parameters.voltage_step == 0):
            analytic_gf_2site(noninteracting_gf.noninteracting_gf, kx[j], ky[i])
    """
    spin_occup_up, spin_occup_down = [0 for i in range(parameters.chain_length)], [0 for i in range(parameters.chain_length)]
    for i in range(0, parameters.chain_length):
        spin_occup_up[i] , spin_occup_down[i] = get_spin_occupation( [ e[i][i] for e in noninteracting_gf.noninteracting_gf ] ,  [ e[i][i] for e in noninteracting_gf.noninteracting_gf] )
    print("The spin up occupation probability is ",  spin_occup_up)  
    """
if __name__=="__main__":#this will only run if it is a script and not a import module
    main()