import scipy.linalg as la
import matplotlib.pyplot as plt
import numpy as np
import math
from dataclasses import dataclass
import time
#some global variables


class HubbardHamiltonian: 
    onsite: float
    gamma: float
    hopping: float
    hubbard_interaction: float
    chain_length: int
    matrix=None
    effective_matrix=None
    Self_energy_left=None
    Self_energy_right=None

    
    def __init__(self, parameters):
        self.onsite = parameters.onsite
        self.hopping = parameters.hopping
        self.chain_length = parameters.chain_length
        self.gamma = parameters.gamma
        self.hubbard_interaction=parameters.hubbard_interaction
      #these are now 1d arrays as I only pass the occupation level for a specific iteration.
            
        self.matrix=create_matrix(self.chain_length)
        self.efffective_matrix=create_matrix(self.chain_length)
        self.Self_energy_left=create_matrix(self.chain_length)
        self.Self_energy_right=create_matrix(self.chain_length)
        
        for i in range(0,self.chain_length):
            self.matrix[i][i]=self.onsite
            
        for i in range(0,self.chain_length-1):
            self.matrix[i][i+1]=self.hopping
            self.matrix[i+1][i]=self.hopping   
                            
        self.Self_energy_right[-1][-1] = -1j * self.gamma      
        self.Self_energy_left[0][0]= -1j * self.gamma
        
        for i in range(0,self.chain_length):
            for j in range(0,self.chain_length):  
                self.efffective_matrix[i][j]=self.matrix[i][j]+self.Self_energy_right[i][j]+self.Self_energy_left[i][j]
    
    def print(self):
        for i in range(0,self.chain_length):
            row_string = " ".join((str(r).rjust(5, " ") for r in self.efffective_matrix[i])) #rjust adds padding, join connects them all
            print(row_string)





@dataclass #creates init function for me
class Parameters:
    onsite: float
    gamma: float
    hopping: float
    chain_length: int
    chemical_potential: float
    temperature: float
    steps: float
    e_upper_bound: float
    e_lower_bound: float
    hubbard_interaction: float


def fermi_function(energy, parameters):
    if(parameters.temperature==0):
        if(energy < parameters.chemical_potential):
            return 1
        else:
            return 0
    else:
        return 1/(1+math.exp((energy-parameters.chemical_potential)/parameters.temperature))
    
def integrating_function(pdos, energy, parameters):
    delta_energy = (parameters.e_upper_bound-parameters.e_lower_bound)/parameters.steps
    result=0
    for r in range(0,parameters.steps):
        result=delta_energy * fermi_function(energy[r], parameters) * pdos[r] +result#pdos=PDOS
    
    return result

def spectral_function_calculator( green_function , parameters):
    spectral_function=create_matrix(parameters.chain_length)
    for i in range(0,parameters.chain_length):
        for j in range(0,parameters.chain_length):
            spectral_function[i][j]=1j*(green_function[i][j]-np.conjugate(green_function[j][i]))  
    return spectral_function
  
    

def create_matrix(size):
    return [[0.0 for x in range(size)] for y in range(size)]
          
def green_function_calculator( hamiltonian , self_energy, parameters, energy):
    inverse_green_function=create_matrix(parameters.chain_length)
    for i in range(0,parameters.chain_length):
           for j in range(0,parameters.chain_length): 
               if(i==j):                
                   inverse_green_function[i][j]=-hamiltonian.efffective_matrix[i][j]+energy-self_energy[i][j]
               else:
                   inverse_green_function[i][j]=-hamiltonian.efffective_matrix[i][j]
        #print(inverse_green_function[i])
    #hamiltonian.print()
    
    return la.inv(inverse_green_function, overwrite_a=False, check_finite=True)    
    

def get_spin_occupation(spectral_function, energy, parameters):
    
    x= 1/(2*np.pi)*integrating_function( spectral_function ,energy, parameters) 
    return x, 1-x

def get_self_consistent_occup(parameters,  energy ):
    gf_hf_up=[create_matrix(parameters.chain_length) for z in range(0,parameters.steps)] 
    gf_hf_down=[create_matrix(parameters.chain_length) for z in range(0,parameters.steps)]
    spectral_function_up=[create_matrix(parameters.chain_length) for z in range(0,parameters.steps)] 
    spectral_function_down=[create_matrix(parameters.chain_length) for z in range(0,parameters.steps)]
    self_energy_up=[create_matrix(parameters.chain_length) for z in range(0,parameters.steps)] 
    self_energy_down=[create_matrix(parameters.chain_length) for z in range(0,parameters.steps)]     
    spin_up_occup , spin_down_occup = [ 0.0 for x in range(0, parameters.chain_length)] , [ 0.0 for x in range(0, parameters.chain_length)]
    #this creates [ [ [0,0,0] , [0,0,0],, [0,0,0] ] , [0,0,0] , [0,0,0],, [0,0,0] ] ... ], ie one chain_length by chain_length 
    # dimesional create_matrix for each energy. The first index in spectral function refers to what energy we are selcting. 
    #the next two indices refer to which enter in our create_matrix we are selecting.    
    n=parameters.chain_length**2*parameters.steps
    differencelist=[0 for i in range(0,2*n)]
    old_green_function=[[[1.0+1j for x in range(parameters.chain_length)] for y in range(parameters.chain_length)] for z in range(0,parameters.steps)] 
    difference=100.0
    hamiltonian_up=HubbardHamiltonian(parameters)
    hamiltonian_down=HubbardHamiltonian(parameters) 
    
    while(difference>0.1):

        for r in range(0,parameters.steps):
            gf_hf_up[r]=green_function_calculator( hamiltonian_up , self_energy_up[r], parameters, energy[r])
            gf_hf_down[r]=green_function_calculator( hamiltonian_down, self_energy_down[r], parameters, energy[r])
            
            spectral_function_up[r]=spectral_function_calculator( gf_hf_up[r] , parameters)
            spectral_function_down[r]=spectral_function_calculator( gf_hf_down[r] , parameters)   
            
        for i in range(0,parameters.chain_length): #this is due to the spin_up_occup being of length chain_length
           spin_up_occup[i] , spin_down_occup[i] = get_spin_occupation([ e[i][i] for e in spectral_function_up], energy, parameters)

        #spin_up_occup.append( 1/(2*np.pi)*integrating_function( [ e[0][0] for e in spectral_function] , energy , parameters) )
        for r in range(0,parameters.steps):
                for i in range(0,parameters.chain_length):
                    self_energy_up[r][i][i]=parameters.hubbard_interaction*spin_down_occup[i]
                    self_energy_down[r][i][i]=parameters.hubbard_interaction*spin_up_occup[i]
                    for j in range(0, parameters.chain_length): #this is due to the spin_up_occup being of length chain_length
                    
                        differencelist[r+i+j]=abs(gf_hf_up[r][i][j].real-old_green_function[r][i][j].real)/abs(old_green_function[r][i][j].real)*100
                        differencelist[n+r+i+j]=abs(gf_hf_up[r][i][j].imag-old_green_function[r][i][j].imag)/abs(old_green_function[r][i][j].imag)*100
                        old_green_function[r][i][j]=gf_hf_up[r][i][j]

        difference=max(differencelist)
        print("The difference is " , difference)


                
        print(spin_up_occup)
    return gf_hf_up, gf_hf_down, spectral_function_up, spectral_function_down, spin_up_occup, spin_down_occup



def main():
    time_start = time.perf_counter()
    onsite, gamma, hopping, chemical_potential, temperature , hubbard_interaction = 1.0 , 2.0 , -1.0 ,0.0, 0.0 , 0.30
    chain_length=1
    
    steps=81 #number of energy points
    e_upper_bound , e_lower_bound = 20.0 , -20.0
    
    spin_up_occup , spin_down_occup = [ 0.0 for x in range(0, chain_length)] , [ 1.0 for x in range(0, chain_length)]

    parameters=Parameters(onsite, gamma, hopping, chain_length, chemical_potential, temperature, steps, e_upper_bound ,e_lower_bound, hubbard_interaction)


    energy=[e_lower_bound+(e_upper_bound-e_lower_bound)/steps*x for x in range(steps)]
    spectral_function_up=[create_matrix(chain_length) for z in range(0,steps)] 
    spectral_function_down=[create_matrix(chain_length) for z in range(0,steps)]
    #this creates [ [ [0,0,0] , [0,0,0],, [0,0,0] ] , [0,0,0] , [0,0,0],, [0,0,0] ] ... ], ie one chain_length by chain_length 
    # dimesional create_matrix for each energy. The first index in spectral function refers to what energy we are selcting. 
    #the next two indices refer to which enter in our create_matrix we are selecting.  
    
    
    gf_hf_up, gf_hf_down, spectral_function_up, spectral_function_down, spin_up_occup, spin_down_occup = get_self_consistent_occup(parameters,  energy )

    magnetisation=[spin_up_occup[i]-spin_down_occup[i] for i in range(0,chain_length)]
    print(" The spin up ocupation is ", spin_up_occup)
    print(" The spin down ocupation is ", spin_down_occup)
    #analytic2=[(2/np.pi)*gamma/((energy[x]-onsite-hubbard_interaction*spin_up_occup[-1]+hubbard_interaction*spin_down_occup[-1]*spin_up_occup[-1])**2+4*gamma**2) for x in range(steps)]   
    print("The magnetisation is ", magnetisation)
    #print(count)
    
    fig = plt.figure()
   

    for i in range(0,chain_length):
        plt.plot(energy, [ e[i][i] for e in spectral_function_up]  , color='blue' ) 
        plt.plot(energy, [ -e[i][i] for e in spectral_function_down], color='red')

    plt.legend(loc='upper right')
    plt.xlabel("energy")
    plt.ylabel("Sepctral Function")  
    plt.show()
    
    time_elapsed = (time.perf_counter() - time_start)
    print(" The time it took the computation for convergence method 3 is" , time_elapsed)
            
if __name__=="__main__":#this will only run if it is a script and not a import module
    main()
 
                

    """
    density_matrix_S=create_matrix(chain_length)   
    for i in range(0,chain_length):
        for j in range(0,chain_length):
            density_matrix_S[i][j]=integrating_function( [ e[i][j] for e in spectral_function] ,energy, parameters)#e is Energy
            print(density_matrix_S[i][j])
        print(" ")
    """

    """
        for i in range(0,chain_length): #this is due to the spin_up_occup being of length chain_length
           x, y = get_spin_occupation([ e[i][i] for e in spectral_function_up], energy, parameters)
           differencelist[i]=abs(spin_up_occup[i]-x)
           spin_up_occup[i]=x
           spin_down_occup[i] = y
    """ 
