import scipy.linalg as la
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import math
from dataclasses import dataclass
#some global variables


class Hamiltonian: 
    onsite: float
    Gamma: float
    hopping: float
    chain_length: int
    matrix=None
    effective_matrix=None
    Self_energy_left=None
    Self_energy_right=None
    
    def __init__(self, parameters):
        self.onsite = parameters.onsite
        self.hopping = parameters.hopping
        self.chain_length = parameters.chain_length
        self.Gamma = parameters.Gamma
        

        self.matrix=[[0.0 for x in range(self.chain_length)] for y in range(self.chain_length)]
        self.efffective_matrix=create_matrix(self.chain_length)
        self.Self_energy_left=create_matrix(self.chain_length)
        self.Self_energy_right=create_matrix(self.chain_length)

        for i in range(0,self.chain_length):
            self.matrix[i][i]=self.onsite
            
        for i in range(0,self.chain_length-1):
            self.matrix[i][i+1]=self.hopping
            self.matrix[i+1][i]=self.hopping   
        
        self.Self_energy_right[self.chain_length-1][self.chain_length-1] = -1j * self.Gamma
        self.Self_energy_left[0][0]= -1j * self.Gamma
        
        for i in range(0,self.chain_length):
            for j in range(0,self.chain_length):  
                self.efffective_matrix[i][j]=self.matrix[i][j]+self.Self_energy_right[i][j]+self.Self_energy_left[i][j]

    
    def print(self):
        for i in range(0,self.chain_length):
            row_string = " ".join((str(r).rjust(5, " ") for r in self.matrix[i])) #rjust adds padding, join connects them all
            print(row_string)

@dataclass #creates init function for me
class Parameters:
    onsite: float
    Gamma: float
    hopping: float
    chain_length: int
    chemical_potential: float
    temperature: float
    steps: float
    e_upper_bound: float
    e_lower_bound: float

def fermi_function(energy, parameters):
    if(parameters.temperature==0):
        if(energy < parameters.chemical_potential):
            return 1
        else:
            return 0
    else:
        return 1/(1+math.exp((energy-parameters.chemical_potential)/parameters.temperature))
    
def integrating_function(pdos, energy, parameters):
    steps=parameters.steps
    e_upper_bound=parameters.e_upper_bound
    e_lower_bound=parameters.e_lower_bound
    delta_energy=(e_upper_bound-e_lower_bound)/steps
    result=0
    for r in range(0,steps):
        result=delta_energy * fermi_function(energy[r], parameters) * pdos[r] +result#pdos=PDOS
    
    return result

def create_matrix(size):
    return [[0.0 for x in range(size)] for y in range(size)]
          
def green_function_calculator( hamiltonian , parameters, energy):
    inverse_green_function=create_matrix(parameters.chain_length)
    for i in range(0,parameters.chain_length):
           for j in range(0,parameters.chain_length): 
               if(i==j):                
                    inverse_green_function[i][j]=-hamiltonian.efffective_matrix[i][j]+energy
               else:
                   inverse_green_function[i][j]=-hamiltonian.efffective_matrix[i][j]
        #print(inverse_green_function[i])
    #hamiltonian.print()
    
    return la.inv(inverse_green_function, overwrite_a=False, check_finite=True)    
    



def main():
    chain_length=1
    steps=2001 #number of energy points
    e_upper_bound = 20.0
    e_lower_bound = -20.0
    Gamma=4.0
    onsite_energy=0.0
    hopping_parameter=1.0
    parameters=Parameters(onsite_energy, Gamma, hopping_parameter, chain_length, 1.0, 0, steps, e_upper_bound ,e_lower_bound)

    hamiltonian=Hamiltonian(parameters)
    #hamiltonian.print()
    A=[ [0 for x in range(steps)] for y in range(chain_length)]
    energy=[e_lower_bound+(e_upper_bound-e_lower_bound)/steps*x for x in range(steps)]
    analytic=[2*Gamma/(np.pi*((energy[x]-onsite_energy)**2+4*Gamma**2)) for x in range(steps)]
    #analytic = [-0.5*(-2*Gamma*(energy[i] - onsite_energy) ** 2 + Gamma * ((energy[i]-onsite_energy) ** 2 - Gamma ** 2 - hopping_parameter**2))/(np.pi * (((energy[i]-onsite_energy) ** 2 - Gamma**2-hopping_parameter**2)**2+4*Gamma**2*(energy[i]-onsite_energy)**2)) for i in range(steps)]

    green_function= create_matrix(chain_length)
    spectral_function=[create_matrix(chain_length) for z in range(0,steps)]
    #this creates [ [ [0,0,0] , [0,0,0],, [0,0,0] ] , [0,0,0] , [0,0,0],, [0,0,0] ] ... ], ie one chain_length by chain_length 
    # dimesional create_matrix for each energy. The first index in spectral function refers to what energy we are selcting. 
    #the next two indices refer to which enter in our create_matrix we are selecting.
    
    for r in range(0,steps):
        green_function=green_function_calculator( hamiltonian , parameters, energy[r])
        for i in range(0,chain_length):
            for j in range(0,chain_length):
                spectral_function[r][i][j]=1j/(2*np.pi)*(green_function[i][j]-np.conjugate(green_function[j][i]))###check
            #print(green_function[i])
            A[i][r]=spectral_function[r][i][i]
    
    density_matrix_S=create_matrix(chain_length)   
    for i in range(0,chain_length):
        for j in range(0,chain_length):
            density_matrix_S[i][j]=integrating_function( [ e[i][j] for e in spectral_function] ,energy, parameters)#e is Energy
            print(density_matrix_S[i][j])
        print(" ")

    fig = plt.figure()
    


    for i in range(0,chain_length):
        color = float(i/chain_length)
        rgb = plt.get_cmap('jet')(color)

        plt.plot(energy, A[i] )        

    plt.plot(energy,analytic, color='tomato')
    plt.legend(loc='upper right')
    plt.xlabel("energy")
    plt.ylabel("Sepctral Function")  
    plt.show()
            
if __name__=="__main__":#this will only run if it is a script and not a import module
    main()