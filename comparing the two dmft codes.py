import scipy.linalg as la
import matplotlib.pyplot as plt
import numpy as np
import math
from dataclasses import dataclass
import time



class HubbardHamiltonian: 
    onsite: float
    gamma: float
    hopping: float
    hubbard_interaction: float
    chain_length: int
    matrix: None
    effective_matrix=None
    self_energy_left=None
    self_energy_right=None


    
    def __init__(self, parameters  ):
        self.onsite , self.hopping , self.chain_length , self.gamma , self.hubbard_interaction = parameters.onsite , parameters.hopping , parameters.chain_length , parameters.gamma , parameters.hubbard_interaction
        self.matrix=create_matrix(self.chain_length)
        self.efffective_matrix=create_matrix(self.chain_length)
        self.self_energy_left=create_matrix(self.chain_length)
        self.self_energy_right=create_matrix(self.chain_length)

        for i in range(0,self.chain_length):
                self.matrix[i][i]=self.onsite

            
        for i in range(0,self.chain_length-1):
            self.matrix[i][i+1]=self.hopping
            self.matrix[i+1][i]=self.hopping   
                            
        self.self_energy_right[-1][-1] = -1j * self.gamma      
        self.self_energy_left[0][0]= -1j * self.gamma
        
        for i in range(0,self.chain_length):
            for j in range(0,self.chain_length):  
                self.efffective_matrix[i][j]=self.matrix[i][j]+self.self_energy_right[i][j]+self.self_energy_left[i][j]
    
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

def integrate( parameters,   gf_1, gf_2, gf_3, r):# in this function, the green functions are 1d arrays in energy. this is becasue we have passed the diagonal component of the green fun( lesser, or retarded) 
    delta_energy = (parameters.e_upper_bound-parameters.e_lower_bound)/parameters.steps
    result=0
        #if i=0, j=0 this corresponds the begining of the energy array, so energies of -20, -20. this means our third green function has an energy of -40 + E[r]. We approximated earlier that
        #the green function is zero outside (-20,20) so E(r)=20 for the integral to be non-zero and hence r=steps. So i+j+r has a minimum value of steps. By a similiar logic, it has a max value of 2*steps.
        # for the range inbetween the index of the third green function is (i+j+r) % steps

    for i in range(0,parameters.steps):
        for j in range(0,parameters.steps):
            if ( ( (i+j+r) >= parameters.steps ) and ( (i+j+j) <= 2*parameters.steps) ):
                
                result=(delta_energy/(2*np.pi))**2 * gf_1[i] * gf_2[j] * gf_3[ (i+j+r)%parameters.steps ] +result

            else:
                result=result
                
    return result

def green_lesser_calculator( parameters , green_function, energy):
    g_lesser=[create_matrix(parameters.chain_length) for z in range(0,parameters.steps)]  
    
    for r in range(0, parameters.steps):
        for i in range(0, parameters.chain_length):
            for j in range(0, parameters.chain_length):
                g_lesser[r][i][j]= -2j*fermi_function( energy[r], parameters)*(green_function[r][i][j]-np.conjugate(green_function[r][j][i]))  
    return g_lesser

def self_energy_calculator(parameters, g_0_up, g_0_down, energy):# this creates the entire energy array at once
    self_energy=[create_matrix(parameters.chain_length) for z in range(0,parameters.steps)] 
    g_lesser_up=[create_matrix(parameters.chain_length) for z in range(0,parameters.steps)]  
    g_lesser_down=[create_matrix(parameters.chain_length) for z in range(0,parameters.steps)]    
    

    g_lesser_up=green_lesser_calculator(parameters,  g_0_up, energy )    
    g_lesser_down=green_lesser_calculator(parameters,  g_0_down, energy )
 

    for r in range(0,parameters.steps):# the are calculating the self energy sigma_{ii}(E) for each discretized energy. To do this we pass the green_fun_{ii} for all energies as we need to integrate over all energies in the integrate function
        for i in range(0, parameters.chain_length):
            self_energy[r][i][i]=  parameters.hubbard_interaction**2*( integrate( parameters, [ e[i][i] for e in g_0_up] , [ e[i][i] for e in g_0_down]  , [ e[i][i] for e in g_lesser_down]   , r )  )
            self_energy[r][i][i]+= parameters.hubbard_interaction**2*( integrate( parameters, [ e[i][i] for e in g_0_up] , [ e[i][i] for e in g_lesser_down] , [ e[i][i] for e in g_lesser_down]  ,r  ) ) 
            self_energy[r][i][i]+= parameters.hubbard_interaction**2*( integrate( parameters, [ e[i][i] for e in g_lesser_up] , [ e[i][i] for e in g_0_down] , [ e[i][i] for e in g_lesser_down]  ,r  ) ) 
            self_energy[r][i][i]+= parameters.hubbard_interaction**2*( integrate( parameters, [ e[i][i] for e in g_lesser_up] , [ e[i][i] for e in g_lesser_down]  , [np.conjugate( e[i][i]) for e in g_0_down]  ,r  ) ) #fix advanced green function

    return self_energy



def get_self_consistent_green_function(parameters, energy):
    gf_int_up=[create_matrix(parameters.chain_length) for z in range(0,parameters.steps)] 
    gf_int_down=[create_matrix(parameters.chain_length) for z in range(0,parameters.steps)]
    spectral_function_up=[create_matrix(parameters.chain_length) for z in range(0,parameters.steps)] 
    spectral_function_down=[create_matrix(parameters.chain_length) for z in range(0,parameters.steps)]
    self_energy_up=[create_matrix(parameters.chain_length) for z in range(0,parameters.steps)] 
    self_energy_down=[create_matrix(parameters.chain_length) for z in range(0,parameters.steps)] 

    spin_up_occup , spin_down_occup = [ 0.0 for x in range(0, parameters.chain_length)] , [ 0.0 for x in range(0, parameters.chain_length)]

    hamiltonian_up=HubbardHamiltonian(parameters)
    hamiltonian_down=HubbardHamiltonian(parameters)
    n=parameters.chain_length**2*parameters.steps
    differencelist=[0 for i in range(0,2*n)]
    old_green_function=[[[1.0+1j for x in range(parameters.chain_length)] for y in range(parameters.chain_length)] for z in range(0,parameters.steps)] 
    difference=100.0
    while (difference>0.1) :


        for r in range(0,parameters.steps):
            gf_int_up[r]=green_function_calculator( hamiltonian_up ,self_energy_up[r] ,  parameters, energy[r])
            gf_int_down[r]=green_function_calculator( hamiltonian_down, self_energy_down[r], parameters, energy[r])
            
            spectral_function_up[r]=spectral_function_calculator( gf_int_up[r] , parameters)
            spectral_function_down[r]=spectral_function_calculator( gf_int_down[r] , parameters)   
        #spin_up_occup.append( 1/(2*np.pi)*integrating_function( [ e[0][0] for e in spectral_function] , energy , parameters) )
        for i in range(0,parameters.chain_length): #this is due to the spin_up_occup being of length chain_length
            spin_up_occup[i] , spin_down_occup[i] = get_spin_occupation([ e[i][i] for e in spectral_function_up], energy, parameters)

        
        self_energy_up=self_energy_calculator(parameters, gf_int_up, gf_int_down, energy )
        self_energy_down=self_energy_calculator(parameters, gf_int_down, gf_int_up, energy )

            
        for r in range(0,parameters.steps):
                for i in range(0,parameters.chain_length):
                    self_energy_up[r][i][i]+=parameters.hubbard_interaction*spin_down_occup[i]
                    self_energy_down[r][i][i]+=parameters.hubbard_interaction*spin_up_occup[i]
                    for j in range(0, parameters.chain_length): #this is due to the spin_up_occup being of length chain_length
                    
                        differencelist[r+i+j]=abs(gf_int_up[r][i][j].real-old_green_function[r][i][j].real)/abs(old_green_function[r][i][j].real)*100
                        differencelist[n+r+i+j]=abs(gf_int_up[r][i][j].imag-old_green_function[r][i][j].imag)/abs(old_green_function[r][i][j].imag)*100
                        old_green_function[r][i][j]=gf_int_up[r][i][j]
                        
        difference=max(differencelist)
        #print("The difference is " , difference)
        #print("The mean difference is ", np.mean(differencelist))
        
    """
    for i in range(0, parameters.chain_length):
        fig = plt.figure()
        
        plt.plot(energy, [e[i][i].imag for e in self_energy_up], color='blue', label='imaginary self energy' ) 
        plt.plot(energy, [e[i][i].real for e in self_energy_up] , color='red' , label='real self energy') 

        plt.legend(loc='upper right')
        plt.xlabel("energy")
        plt.ylabel("Self Energy")  
        plt.show()
    """
    #print("The spin up occupaton probability is ", spin_up_occup)
    return gf_int_up, gf_int_down, spectral_function_up, spectral_function_down, spin_up_occup, spin_down_occup

    

def green_lesser_calculator1( parameters , green_function, energy):
    g_lesser=[create_matrix(1) for z in range(0,parameters.steps)]  
    for r in range(0, parameters.steps):
        for i in range(0, 1):
            for j in range(0, 1):
                g_lesser[r][i][j]= -2j*fermi_function( energy[r], parameters)*(green_function[r][i][j]-np.conjugate(green_function[r][j][i]))  
    return g_lesser

def self_energy_calculator1(parameters, g_0_up, g_0_down, energy):# this creates the entire energy array at once
    self_energy=[create_matrix(parameters.chain_length) for z in range(0,parameters.steps)] 
    g_lesser_up=[create_matrix(parameters.chain_length) for z in range(0,parameters.steps)]  
    g_lesser_down=[create_matrix(parameters.chain_length) for z in range(0,parameters.steps)]    
    

    g_lesser_up=green_lesser_calculator1(parameters,  g_0_up, energy )    
    g_lesser_down=green_lesser_calculator1(parameters,  g_0_down, energy )
 

    for r in range(0,parameters.steps):# the are calculating the self energy sigma_{ii}(E) for each discretized energy. To do this we pass the green_fun_{ii} for all energies as we need to integrate over all energies in the integrate function
        for i in range(0, 1):
            self_energy[r][i][i]=  parameters.hubbard_interaction**2*( integrate( parameters, [ e[i][i] for e in g_0_up] , [ e[i][i] for e in g_0_down]  , [ e[i][i] for e in g_lesser_down]   , r )  )
            self_energy[r][i][i]+= parameters.hubbard_interaction**2*( integrate( parameters, [ e[i][i] for e in g_0_up] , [ e[i][i] for e in g_lesser_down] , [ e[i][i] for e in g_lesser_down]  ,r  ) ) 
            self_energy[r][i][i]+= parameters.hubbard_interaction**2*( integrate( parameters, [ e[i][i] for e in g_lesser_up] , [ e[i][i] for e in g_0_down] , [ e[i][i] for e in g_lesser_down]  ,r  ) ) 
            self_energy[r][i][i]+= parameters.hubbard_interaction**2*( integrate( parameters, [ e[i][i] for e in g_lesser_up] , [ e[i][i] for e in g_lesser_down]  , [np.conjugate( e[i][i]) for e in g_0_down]  ,r  ) ) #fix advanced green function

    return self_energy


def spectral_function_calculator( green_function , parameters):
    spectral_function=create_matrix(parameters.chain_length)
    for i in range(0,parameters.chain_length):
        for j in range(0,parameters.chain_length):
            spectral_function[i][j]=1j*(green_function[i][j]-np.conjugate(green_function[j][i]))  
    return spectral_function
  
  
    

def create_matrix(size):
    return [[0.0 for x in range(size)] for y in range(size)]
          
def green_function_calculator( hamiltonian , self_energy , parameters, energy):
    inverse_green_function=create_matrix(parameters.chain_length)
    for i in range(0,parameters.chain_length):
           for j in range(0,parameters.chain_length): 
               if(i==j):                
                   inverse_green_function[i][j]=-hamiltonian.efffective_matrix[i][j]-self_energy[i][j] +energy
               else:
                   inverse_green_function[i][j]=-hamiltonian.efffective_matrix[i][j]
        #print(inverse_green_function[i])
    #hamiltonian.print()
    
    return la.inv(inverse_green_function, overwrite_a=False, check_finite=True)    
    

def get_spin_occupation(spectral_function, energy, parameters):
    
    x= 1/(2*np.pi)*integrating_function( spectral_function ,energy, parameters) 
    return x, 1-x

def inner_dmft(parameters, gf_int_up, gf_int_down, energy):
    g_local_up=[create_matrix(1) for i in range(parameters.steps)]
    g_local_down=[create_matrix(1) for i in range(parameters.steps)]
    local_spectral_up=[create_matrix(1) for i in range(parameters.steps)]
    self_energy_up=[create_matrix(parameters.chain_length) for z in range(0,parameters.steps)] 
    self_energy_down=[create_matrix(parameters.chain_length) for z in range(0,parameters.steps)] 
    local_sigma_up=[create_matrix(1) for i in range(parameters.steps)]
    spin_up_occup , spin_down_occup = [ 0.0 for x in range(0, parameters.chain_length)] , [ 0.0 for x in range(0, parameters.chain_length)]
    local_sigma_down=[create_matrix(1) for i in range(parameters.steps)]
  
    hamiltonian=HubbardHamiltonian(parameters)
    g_initial=[create_matrix(parameters.chain_length) for z in range(0,parameters.steps)] 
    for r in range(0,parameters.steps):
        g_initial[r]=green_function_calculator( hamiltonian ,self_energy_up[r] ,  parameters, energy[r])
 
        
    g_initial_up=[r for z in range(0,parameters.steps)]
    g_initial_down=[r for z in range(0,parameters.steps)]
    n=parameters.chain_length*parameters.steps       
    differencelist=[0 for i in range(0,2*n)]    
    for i in range(0, parameters.chain_length):
              
        for r in range(0,parameters.steps):
            g_local_up[r][0][0]=gf_int_up[r][i][i]
            g_local_down[r][0][0]=gf_int_down[r][i][i]
            

                                
        old_green_function=[0 for z in range(0,parameters.steps)] 
        difference=100.0
        while(difference>0.001):
            #print(g_local_up)
            for r in range(0,parameters.steps):
                local_spectral_up[r][0][0]=1j*(g_local_up[r][0][0]-np.conjugate(g_local_up[r][0][0]))

            #print(local_spectral_up)
            local_spin_up , local_spin_down = get_spin_occupation([ e[0][0] for e in local_spectral_up], energy, parameters)
            #print("The spin occupancy is ", local_spin_up, " atom ", i)
            
            #(g_local_up)
            local_sigma_up=self_energy_calculator1(parameters, g_local_up, g_local_down, energy )
            local_sigma_down=self_energy_calculator1(parameters, g_local_down, g_local_up, energy )
            
            for r in range(0,parameters.steps):
                 local_sigma_up[r][0][0]+=parameters.hubbard_interaction*local_spin_down
                 local_sigma_down[r][0][0]+=parameters.hubbard_interaction*local_spin_up
                 
                 g_initial_up[r]=1/((1/g_local_up[r][0][0])+local_sigma_up[r][0][0])
                 g_initial_down[r]=1/((1/g_local_down[r][0][0])+local_sigma_down[r][0][0])
                 
                 g_local_up[r][0][0]=1/((1/g_initial_up[r])-local_sigma_up[r][0][0])
                 g_local_down[r][0][0]=1/((1/g_initial_down[r])-local_sigma_down[r][0][0])  
                 
            for r in range(0,parameters.steps):
                        differencelist[r]=abs(g_local_up[r][0][0].real-old_green_function[r].real)
                        differencelist[n+r]=abs(g_local_up[r][0][0].imag-old_green_function[r].imag)
                        old_green_function[r]=g_local_up[r][0][0]

            difference=max(differencelist)
            #print("The inner difference is " , difference)
            #print("The inner  mean difference is ", np.mean(differencelist)) 
            
        #print(" ")
        for r in range(0,parameters.steps):
            self_energy_up[r][i][i]=local_sigma_up[r][0][0]
            self_energy_down[r][i][i]=local_sigma_down[r][0][0]            
        spin_up_occup[i]=local_spin_up
        spin_down_occup[i]=local_spin_down
    
    return self_energy_up, self_energy_down, spin_up_occup , spin_down_occup

def gf_dmft(parameters, energy):
    gf_int_up=[create_matrix(parameters.chain_length) for z in range(0,parameters.steps)] 
    gf_int_down=[create_matrix(parameters.chain_length) for z in range(0,parameters.steps)]
    spectral_function_up=[create_matrix(parameters.chain_length) for z in range(0,parameters.steps)] 
    spectral_function_down=[create_matrix(parameters.chain_length) for z in range(0,parameters.steps)]

    self_energy_up=[create_matrix(parameters.chain_length) for z in range(0,parameters.steps)] 
    self_energy_down=[create_matrix(parameters.chain_length) for z in range(0,parameters.steps)] 

    spin_up_occup , spin_down_occup = [ 0.0 for x in range(0, parameters.chain_length)] , [ 0.0 for x in range(0, parameters.chain_length)]

    hamiltonian_up=HubbardHamiltonian(parameters)
    hamiltonian_down=HubbardHamiltonian(parameters)
    n=parameters.chain_length**2*parameters.steps
    differencelist=[0 for i in range(0,2*n)]
    old_green_function=[[[1.0+1j for x in range(parameters.chain_length)] for y in range(parameters.chain_length)] for z in range(0,parameters.steps)] 
    difference=100.0
    while (difference>0.001) :


        for r in range(0,parameters.steps):
            gf_int_up[r]=green_function_calculator( hamiltonian_up ,self_energy_up[r] ,  parameters, energy[r])
            gf_int_down[r]=green_function_calculator( hamiltonian_down, self_energy_down[r], parameters, energy[r])

        self_energy_up, self_energy_down, spin_up_occup , spin_down_occup =inner_dmft(parameters, gf_int_up, gf_int_down, energy)

        #print( spin_up_occup)
        for r in range(0,parameters.steps):
                for i in range(0,parameters.chain_length):
                    for j in range(0, parameters.chain_length): #this is due to the spin_up_occup being of length chain_length
                    
                        differencelist[r+i+j]=abs(gf_int_up[r][i][j].real-old_green_function[r][i][j].real)
                        differencelist[n+r+i+j]=abs(gf_int_up[r][i][j].imag-old_green_function[r][i][j].imag)
                        old_green_function[r][i][j]=gf_int_up[r][i][j]
                        
                        
        difference=max(differencelist)
        #print("The difference is " , difference)
        #print("The mean difference is ", np.mean(differencelist))
        
    for r in range(0,parameters.steps):
        spectral_function_up[r]=spectral_function_calculator(gf_int_up[r], parameters)
        spectral_function_down[r]=spectral_function_calculator(gf_int_down[r], parameters)    
    """
    for i in range(0, parameters.chain_length):
        fig = plt.figure()
        
        plt.plot(energy, [e[i][i].imag for e in self_energy_up], color='blue', label='imaginary self energy' ) 
        plt.plot(energy, [e[i][i].real for e in self_energy_up] , color='red' , label='real self energy') 

        plt.legend(loc='upper right')
        plt.xlabel("energy")
        plt.ylabel("Self Energy")  
        plt.show()
    """
    #print("The spin up occupaton probability is ", spin_up_occup)
    return gf_int_up, gf_int_down, spectral_function_up, spectral_function_down, spin_up_occup, spin_down_occup

    

def main():
    time_start = time.perf_counter()
    onsite, gamma, hopping, chemical_potential, temperature , hubbard_interaction = 1.0 , 2.0 , -1.0 ,0.0, 0.0 , 0.3
    chain_length=3
    steps=81 #number of energy points
    e_upper_bound , e_lower_bound = 20.0 , -20.0
    
    spin_up_occup , spin_down_occup = [ 0.0 for x in range(0, chain_length)] , [ 1.0 for x in range(0, chain_length)]

    parameters=Parameters(onsite, gamma, hopping, chain_length, chemical_potential, temperature, steps, e_upper_bound ,e_lower_bound, hubbard_interaction)


    energy=[e_lower_bound+(e_upper_bound-e_lower_bound)/steps*x for x in range(steps)]

    #this creates [ [ [0,0,0] , [0,0,0],, [0,0,0] ] , [0,0,0] , [0,0,0],, [0,0,0] ] ... ], ie one chain_length by chain_length 
    # dimesional create_matrix for each energy. The first index in spectral function refers to what energy we are selcting. 
    #the next two indices refer to which enter in our create_matrix we are selecting.    
    green_function_up1, green_function_down1, spectral_function_up1, spectral_function_down1, spin_up_occup1, spin_down_occup1 = gf_dmft(parameters,  energy )



    magnetisation=[spin_up_occup[i]-spin_down_occup[i] for i in range(0,chain_length)]
    #analytic2=[(2/np.pi)*gamma/((energy[x]-onsite-hubbard_interaction*spin_up_occup[-1]+hubbard_interaction*spin_down_occup[-1]*spin_up_occup[-1])**2+4*gamma**2) for x in range(steps)]   

    #print(count)
    
    green_function_up2, green_function_down2, spectral_function_up2, spectral_function_down2, spin_up_occup2, spin_down_occup2 = get_self_consistent_green_function(parameters,  energy )

    magnetisation=[spin_up_occup[i]-spin_down_occup[i] for i in range(0,chain_length)]
    #analytic2=[(2/np.pi)*gamma/((energy[x]-onsite-hubbard_interaction*spin_up_occup[-1]+hubbard_interaction*spin_down_occup[-1]*spin_up_occup[-1])**2+4*gamma**2) for x in range(steps)]   
    print("The spin up occupation for 1 loop is ", spin_up_occup2)
    print("The spin up occupation for 2 loops is ", spin_up_occup1)
    #print(count)
    

    fig = plt.figure()
   
    for i in range(0,chain_length):
        plt.plot(energy, [ e[i][i] for e in spectral_function_up1]  , color='blue' , label='2 loops') 
        plt.plot(energy, [ e[i][i] for e in spectral_function_up2]  , color='green' ,label='1 loop' ) 
        #plt.plot(energy, [ -e[i][i] for e in spectral_function_down], color='red')
        #plt.plot(energy, dos_spin_up[i] , color='blue', label='spin up DOS' ) 
        #plt.plot(energy, dos_spin_down[i], color='red', label='spin down DOS')
#
   # plt.plot(energy,analytic, color='tomato')
    #plt.plot(energy,analytic2, color='green')
    plt.legend(loc='upper right')
    plt.xlabel("energy")
    plt.ylabel("Sepctral Function")  
    plt.show()


    for i in range(0,chain_length):
        plt.plot(energy, [ e[i][i] for e in green_function_up1]  , color='blue' , label='2 loops') 
        plt.plot(energy, [ e[i][i] for e in green_function_up2]  , color='green' ,label='1 loop' ) 
        #plt.plot(energy, [ -e[i][i] for e in spectral_function_down], color='red')
        #plt.plot(energy, dos_spin_up[i] , color='blue', label='spin up DOS' ) 
        #plt.plot(energy, dos_spin_down[i], color='red', label='spin down DOS')
#
   # plt.plot(energy,analytic, color='tomato')
    #plt.plot(energy,analytic2, color='green')
    plt.legend(loc='upper right')
    plt.xlabel("energy")
    plt.ylabel("Real GF")  
    plt.show()
    
    time_elapsed = (time.perf_counter() - time_start)
    print(" The time it took the computation for convergence method 3 is" , time_elapsed)
            
if __name__=="__main__":#this will only run if it is a script and not a import module
    main()







