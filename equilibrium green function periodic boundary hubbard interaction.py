import scipy.linalg as la
import matplotlib.pyplot as plt
import numpy as np
import math
from dataclasses import dataclass
#some global variables


class PeriodicBCHubbardHamiltonian: 
    onsite: float
    gamma: float
    hopping_x: float
    hopping_y: float
    hubbard_interaction: float
    spin_occup: None
    chain_length_x: int
    chain_length_y: int
    matrix=None
    effective_matrix= None
    Self_energy_left=None
    Self_energy_right=None ##markus
    k_y: int
    
    
    
    
    def __init__(self, parameters, _spin_occup , _k_y ):
        self.onsite = parameters.onsite
        self.hopping_x , self.hopping_y = parameters.hopping_x , parameters.hopping_y
        self.chain_length_x , self.chain_length_y = parameters.chain_length_x , parameters.chain_length_y
        self.gamma , self.hubbard_interaction  = parameters.gamma , parameters.hubbard_interaction
        self.spin_occup  = _spin_occup #these are now 1d arrays as I only pass the occupation level for a specific iteration.
        self.k_y= _k_y
             
        self.matrix=create_matrix(self.chain_length_x)
        self.effective_matrix=create_matrix(self.chain_length_x)
        self.Self_energy_left=create_matrix(self.chain_length_x)
        self.Self_energy_right=create_matrix(self.chain_length_x)
        
        
        if(self.chain_length_y != 1):           
            for i in range(0,self.chain_length_x):
                self.matrix[i][i]=self.onsite+self.hubbard_interaction*self.spin_occup[i]+self.hopping_y*2*np.cos(self.k_y)
        elif(self.chain_length_y == 1):  
            for i in range(0,self.chain_length_x):
                self.matrix[i][i]=self.onsite+self.hubbard_interaction*self.spin_occup[i]
          
        for i in range(0,self.chain_length_x-1):
            self.matrix[i][i+1]=self.hopping_x
            self.matrix[i+1][i]=self.hopping_x  
                            
        self.Self_energy_right[-1][-1] = -1j * self.gamma      
        self.Self_energy_left[0][0]= -1j * self.gamma
        
        for i in range(0,self.chain_length_x):
            for j in range(0,self.chain_length_x):  
                self.effective_matrix[i][j]=self.matrix[i][j]+self.Self_energy_right[i][j]+self.Self_energy_left[i][j]
    
    def print(self):
        for i in range(0,self.chain_length_x):
            row_string = " ".join((str(r).rjust(5, " ") for r in self.effective_matrix[i])) #rjust adds padding, join connects them all
            print(row_string)




@dataclass #creates init function for me
class Parameters:
    onsite: float
    gamma: float
    hopping_x: float
    hopping_y: float
    chain_length_x: int
    chain_length_y: int
    chemical_potential: float
    temperature: float
    steps: float
    e_upper_bound: float
    e_lower_bound: float
    hubbard_interaction: float

#this function is completely useless for this exercise but could be handy in the future.
"""
def vector(i,chain_length_x ): # this function assigns a number a specific vector which corresponds to its position in the system
    position=[0,0]
    position[0]=i%chain_length_x+1
    position[1]=1
    idk=i
    while(idk>0):
        idk=idk-chain_length_x
        if(idk>=0):
            position[1]=position[1]+1
    #print(position , i)
    return position
"""
def fermi_function(energy, parameters):
    if(parameters.temperature==0):
        if(energy < parameters.chemical_potential):
            return 1
        else:
            return 0
    else:
        return 1/(1+math.exp((energy-parameters.chemical_potential)/parameters.temperature))


def spectral_function_calculator( green_function , parameters):
    spectral_function=create_matrix(parameters.chain_length_x)
    for i in range(0,parameters.chain_length_x):
        for j in range(0,parameters.chain_length_x):
            spectral_function[i][j]=1j*(green_function[i][j]-np.conjugate(green_function[j][i]))  
    return spectral_function    

def integrating_function(pdos, energy, parameters):
    delta_energy = (parameters.e_upper_bound-parameters.e_lower_bound)/parameters.steps
    result=0
    for r in range(0,parameters.steps):
        result=delta_energy * fermi_function(energy[r], parameters) * pdos[r] +result#pdos=PDOS
    
    return result

def create_matrix(size):
    return [[0.0 for x in range(size)] for y in range(size)]
          
def green_function_calculator( hamiltonian , parameters, energy):
    inverse_green_function=create_matrix(parameters.chain_length_x)
    for i in range(0,parameters.chain_length_x):
           for j in range(0,parameters.chain_length_x): 
               if(i==j):                
                   inverse_green_function[i][j]=-hamiltonian.effective_matrix[i][j]+energy
               else:
                   inverse_green_function[i][j]=-hamiltonian.effective_matrix[i][j]
        #print(inverse_green_function[i])
    #hamiltonian.print()
    
    return la.inv(inverse_green_function, overwrite_a=False, check_finite=True)    
    

def get_spin_occupation(spectral_function, energy, parameters):
    
    x= 1/(2*np.pi)*integrating_function( spectral_function ,energy, parameters) 
    return x, 1-x

def main():
    onsite, gamma, hopping_x , hopping_y , chemical_potential, temperature , hubbard_interaction = 1.0 , 2.0 , -1.0 ,-1.0 ,0.0, 0.0 , 0.3
    chain_length_x,chain_length_y= 3, 1
    steps=81 #number of energy points
    e_upper_bound , e_lower_bound = 20.0 , -20.0
    k_y=[ 2*np.pi*m/chain_length_y for m in range(0,chain_length_y)]
    
    spin_up_occup , spin_down_occup = [ 0.0 for x in range(0, chain_length_x)] , [ 1.0 for x in range(0, chain_length_x)]
    
    difference =1.0 
    parameters=Parameters(onsite, gamma, hopping_x , hopping_y , chain_length_x, chain_length_y , chemical_potential, temperature, steps, e_upper_bound ,e_lower_bound, hubbard_interaction)
    energy=[e_lower_bound+(e_upper_bound-e_lower_bound)/steps*x for x in range(steps)]

    green_function_k_up= create_matrix(chain_length_x)
    green_function_k_down= create_matrix(chain_length_x)
    spectral_function_up=[create_matrix(chain_length_x) for z in range(0,steps)] 
    spectral_function_down=[create_matrix(chain_length_x) for z in range(0,steps)]   
    #this creates [ [ [0,0,0] , [0,0,0],, [0,0,0] ] , [0,0,0] , [0,0,0],, [0,0,0] ] ... ], ie one chain_length by chain_length 
    # dimesional create_matrix for each energy. The first index in spectral function refers to what energy we are selcting. 
    #the next two indices refer to which enter in our create_matrix we are selecting.  
    
    dos_spin_up=[ [0 for x in range(steps)] for y in range(chain_length_x)]
    dos_spin_down=[ [0 for x in range(steps)] for y in range(chain_length_x)]



    while(difference>0.00001):
        green_function_up= [create_matrix(chain_length_x) for z in range(0,steps)]   
        green_function_down= [create_matrix(chain_length_x) for z in range(0,steps)]   
        for r in range(0,steps):

            #print(" ")
            for t in range(0, chain_length_y):
                hamiltonian_up=PeriodicBCHubbardHamiltonian(parameters, spin_down_occup, k_y[t])
                hamiltonian_down=PeriodicBCHubbardHamiltonian(parameters, spin_up_occup, k_y[t])

                green_function_k_up=green_function_calculator( hamiltonian_up , parameters, energy[r])
                green_function_k_down=green_function_calculator( hamiltonian_down , parameters, energy[r])
                
                for i in range(0,chain_length_x):
                    for j in range(0,chain_length_x):
                        green_function_up[r][i][j] = (1/chain_length_y)*green_function_k_up[i][j] + green_function_up[r][i][j]
                        green_function_down[r][i][j] = (1/chain_length_y)*green_function_k_down[i][j] + green_function_down[r][i][j]       
            
            spectral_function_up[r]=spectral_function_calculator( green_function_up[r] , parameters)
            spectral_function_down[r]=spectral_function_calculator( green_function_down[r] , parameters)
            
            for i in range(0,chain_length_x):
                dos_spin_up[i][r]=spectral_function_up[r][i][i]
                dos_spin_down[i][r]=spectral_function_down[r][i][i]
#print(green_function)

        print("did a loop")
        differencelist=[0 for i in range(0,chain_length_x)]
        for i in range(0,chain_length_x): #this is due to the spin_up_occup being of length chain_length
           x, y = get_spin_occupation([ e[i][i] for e in spectral_function_up], energy, parameters)
           differencelist[i]=abs(spin_up_occup[i]-x)
           spin_up_occup[i]=x
           spin_down_occup[i] = y
        
        
        difference=max(differencelist)
        #hamiltonian_up.print()
        #hamiltonian_down.print()
        print(spin_down_occup)
        
        
        """
    density_matrix_S=create_matrix(chain_length)   
    for i in range(0,chain_length):
        for j in range(0,chain_length):
            density_matrix_S[i][j]=integrating_function( [ e[i][j] for e in spectral_function] ,energy, parameters)#e is Energy
            print(density_matrix_S[i][j])
        print(" ")
        """

        #spin_up_occup.append( 1/(2*np.pi)*integrating_function( [ e[0][0] for e in spectral_function] , energy , parameters) )
    #difference=abs(old_spin_up-spin_up_occup)

        #print(spin_up_occup[count], spin_down_occup[count] , count, difference)

    #analytic=[(2/np.pi)*gamma/((energy[x]-onsite-hubbard_interaction*spin_down_occup[-1]+hubbard_interaction*spin_down_occup[-1]*spin_up_occup[-1])**2+4*gamma**2) for x in range(steps)]   
    magnetisation=[spin_up_occup[i]-spin_down_occup[i] for i in range(0,chain_length_x)]
    #analytic2=[(2/np.pi)*gamma/((energy[x]-onsite-hubbard_interaction*spin_up_occup[-1]+hubbard_interaction*spin_down_occup[-1]*spin_up_occup[-1])**2+4*gamma**2) for x in range(steps)]   
    print("The magnetisation is ", magnetisation)
    #print(count)
  
    fig = plt.figure()
    


    for i in range(0,chain_length_x):
        color = float(i/chain_length_x)
        rgb = plt.get_cmap('jet')(color)
    #print(K,Eigenvalues[i])

        plt.plot(energy, dos_spin_up[i] , color='blue' ) 
        plt.plot(energy, dos_spin_down[i], color='red')
#
   # plt.plot(energy,analytic, color='tomato')
    #plt.plot(energy,analytic2, color='green')
    plt.legend(loc='upper right')
    plt.xlabel("energy")
    plt.ylabel("Sepctral Function")  
    plt.show()
            
if __name__=="__main__":#this will only run if it is a script and not a import module
    main()