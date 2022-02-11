import scipy.linalg as la
import matplotlib.pyplot as plt
import numpy as np
import math
from dataclasses import dataclass
import time

class StaticNumber:
     i = 0
     
     def add_one(self):
         self.i+=1

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
        self.efffective_matrix=[create_matrix(self.chain_length) for r in range(parameters.steps)]
        self.self_energy_left=[create_matrix(self.chain_length) for r in range(parameters.steps)]
        self.self_energy_right=[create_matrix(self.chain_length)for r in range(parameters.steps)]
        

        self.self_energy_left , self.self_energy_right = self.embedding_self_energy(parameters)
        
        
         
        for r in range(0,parameters.steps):
            for i in range(0,self.chain_length-1):
                self.matrix[i][i+1]=self.hopping
                self.matrix[i+1][i]=self.hopping 
            for i in range(0,self.chain_length):
                self.matrix[i][i]=self.onsite
                for j in range(0,self.chain_length):  
                    self.efffective_matrix[r][i][j]=self.matrix[i][j] + self.self_energy_right[r][i][j] + self.self_energy_left[r][i][j]

    def embedding_self_energy(self , parameters):
        se_emb_l=[create_matrix(self.chain_length) for r in range(parameters.steps)]
        se_emb_r=[create_matrix(self.chain_length)for r in range(parameters.steps)]
        
        f= open(r"C:\Users\user\Desktop\Green function code\Green's Function\embedding_self_energy.txt", "r")
        lines = f.read().split(',')  
        for r in range(0,parameters.steps):  
            se_emb_r[r][-1][-1]=float(lines[3+r*5])+1j*float(lines[4+r*5])
            se_emb_l[r][0][0]=float(lines[1+r*5])+1j*float(lines[2+r*5])  
        f.close()
        
        return se_emb_l , se_emb_r
    
    def plot_embedding_self_energy( self , energy ):
            plt.plot(energy, [ e[-1][-1].real for e in self.self_energy_right]  , color='blue', label='real self energy' ) 
            plt.plot(energy, [ e[-1][-1].imag for e in self.self_energy_right], color='red', label='imaginary self energy')
            plt.title("embedding self energy")
   # plt.plot(energy,analytic, color='tomato')
    #plt.plot(energy,analytic2, color='green')
            plt.legend(loc='upper right')
            plt.xlabel("energy")
            plt.ylabel("embedding self energy")  
            plt.show()
            
            
    
    def print(self, num):
        for i in range(0,self.chain_length):
            row_string = " ".join((str(r).rjust(5, " ") for r in self.efffective_matrix[num][i])) #rjust adds padding, join connects them all
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

def integrate( parameters,   gf_1, gf_2, gf_3, r):# in this function, the green functions are 1d arrays in energy. this is becasue we have passed the diagonal component of the green fun( lesser, or retarded) 
    delta_energy = (parameters.e_upper_bound-parameters.e_lower_bound)/parameters.steps
    result=0
    energy = [ parameters.e_lower_bound + (parameters.e_upper_bound - parameters.e_lower_bound)/parameters.steps*x for x in range(parameters.steps)]    
        #if i=0, j=0 this corresponds the begining of the energy array, so energies of -20, -20. this means our third green function has an energy of -40 + E[r]. We approximated earlier that
        #the green function is zero outside (-20,20) so E(r)=20 for the integral to be non-zero and hence r=steps. So i+j+r has a minimum value of steps. By a similiar logic, it has a max value of 2*steps.
        # for the range inbetween the index of the third green function is (i+j+r) % steps
    for i in range( 0  ,parameters.steps ):
        for j in range( 0, parameters.steps ):
            if ( ( (energy[i] + energy[j] - energy[r] ) >=  parameters.e_lower_bound ) and ( (energy[i] + energy[j] - energy[r] ) <= parameters.e_lower_bound )   ):             
                result = ( delta_energy / ( 2 * np.pi))**2 * gf_1[i] * gf_2[j] * gf_3[ (i+j-r) ] +result
            else:
                result = result    
              
    return result

def green_lesser_calculator( parameters , green_function, energy):
    g_lesser=[create_matrix(1) for z in range(0,parameters.steps)]  
    for r in range(0, parameters.steps):
        for i in range(0, 1):
            for j in range(0, 1):
                g_lesser[r][i][j]= fermi_function( energy[r], parameters)*(green_function[r][i][j]-np.conjugate(green_function[r][j][i]))  
    return g_lesser

def self_energy_calculator(parameters, g_0_up, g_0_down, energy):# this creates the entire energy array at once
    self_energy=[create_matrix(parameters.chain_length) for z in range(0,parameters.steps)]   
    g_lesser_up=green_lesser_calculator(parameters,  g_0_up, energy )     
    g_lesser_down=green_lesser_calculator(parameters,  g_0_down, energy )
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
          
def green_function_calculator( hamiltonian , self_energy , parameters, energy, energy_step):
    inverse_green_function=create_matrix(parameters.chain_length)
    for i in range(0,parameters.chain_length):
           for j in range(0,parameters.chain_length): 
               if(i==j):                
                   inverse_green_function[i][j] = - hamiltonian.efffective_matrix[energy_step][i][j] - self_energy[i][j] + energy
               else:
                   inverse_green_function[i][j] = - hamiltonian.efffective_matrix[energy_step][i][j]
        #print(inverse_green_function[i])
    #hamiltonian.print()
    
    return la.inv(inverse_green_function, overwrite_a=False, check_finite=True)    
    
def get_spin_occupation(spectral_function, energy, parameters):
    delta_energy = (parameters.e_upper_bound-parameters.e_lower_bound)/parameters.steps
    result=0
    for r in range(0,parameters.steps):
        result = (delta_energy) * fermi_function(energy[r], parameters) * spectral_function[r] +result#pdos=PDOS

    x= 1/(np.pi)*result 
    return x, 1-x

def inner_dmft(parameters, gf_int_up, gf_int_down, energy): #this solves the impurity problem self consistently
    g_local_up=[create_matrix(1) for i in range(parameters.steps)]#computationally this is a pointer to a pointer which contain g_local. This must be inefficient computationally.
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
        g_initial[r]=green_function_calculator( hamiltonian ,self_energy_up[r] ,  parameters, energy[r] ,r)
 
        
    g_initial_up=[r for z in range(0,parameters.steps)]
    g_initial_down=[r for z in range(0,parameters.steps)]
    n=parameters.chain_length*parameters.steps       
    differencelist=[0 for i in range(0,2*n)]    
    for i in range(0, parameters.chain_length):
              
        for r in range(0,parameters.steps):#this sets the impurity green function to the local lattice green function for each lattice site(the i for loop)
            g_local_up[r][0][0]=gf_int_up[r][i][i]
            g_local_down[r][0][0]=gf_int_down[r][i][i]
            

                                
        old_green_function=[0 for z in range(0,parameters.steps)] 
        difference=100.0
        while(difference>0.0001):#this is solving the impurity problem self consistently which in principle should be correct
            #print(g_local_up)
            #to not solve the impurity problem self consistently we should set difference =0 so only 1 while loop occurs
            for r in range(0,parameters.steps):
                local_spectral_up[r][0][0]=1j*(g_local_up[r][0][0]-np.conjugate(g_local_up[r][0][0]))

            #print(local_spectral_up)
            local_spin_up , local_spin_down = get_spin_occupation([ e[0][0] for e in local_spectral_up], energy, parameters)
            #print("The spin occupancy is ", local_spin_up, " atom ", i)
            
            #(g_local_up)
            local_sigma_up=self_energy_calculator(parameters, g_local_up, g_local_down, energy )
            local_sigma_down=self_energy_calculator(parameters, g_local_down, g_local_up, energy )
            
            for r in range(0,parameters.steps):
                 local_sigma_up[r][0][0]+=parameters.hubbard_interaction*local_spin_down
                 local_sigma_down[r][0][0]+=parameters.hubbard_interaction*local_spin_up
                 
                 g_initial_up[r]=1/((1/g_local_up[r][0][0])+local_sigma_up[r][0][0])# this is getting the new dynamical mean field
                 g_initial_down[r]=1/((1/g_local_down[r][0][0])+local_sigma_down[r][0][0])
                 
                 g_local_up[r][0][0]=1/((1/g_initial_up[r])-local_sigma_up[r][0][0])
                 g_local_down[r][0][0]=1/((1/g_initial_down[r])-local_sigma_down[r][0][0])  
                 
            for r in range(0,parameters.steps):
                        differencelist[r]=abs(g_local_up[r][0][0].real-old_green_function[r].real)
                        differencelist[n+r]=abs(g_local_up[r][0][0].imag-old_green_function[r].imag)
                        old_green_function[r]=g_local_up[r][0][0]

            #difference=max(differencelist)
            difference=0
            #print("The inner difference is " , difference)
            #print("The inner  mean difference is ", np.mean(differencelist)) 

        for r in range(0,parameters.steps): #this then returns a diagonal self energy
            self_energy_up[r][i][i]=local_sigma_up[r][0][0]
            self_energy_down[r][i][i]=local_sigma_down[r][0][0]            
        spin_up_occup[i]=local_spin_up
        spin_down_occup[i]=local_spin_down
    
    return self_energy_up, self_energy_down, spin_up_occup , spin_down_occup


    


def gf_dmft(parameters, energy): # this function does not solve the impurity green function self consistently
    gf_int_up=[create_matrix(parameters.chain_length) for z in range(0,parameters.steps)] 
    gf_int_down=[create_matrix(parameters.chain_length) for z in range(0,parameters.steps)]
    spectral_function_up=[create_matrix(parameters.chain_length) for z in range(0,parameters.steps)] 
    spectral_function_down=[create_matrix(parameters.chain_length) for z in range(0,parameters.steps)]

    self_energy_up=[create_matrix(parameters.chain_length) for z in range(0,parameters.steps)] 
    self_energy_down=[create_matrix(parameters.chain_length) for z in range(0,parameters.steps)] 

    spin_up_occup , spin_down_occup = [ 0.0 for x in range(0, parameters.chain_length)] , [ 0.0 for x in range(0, parameters.chain_length)]

    hamiltonian=HubbardHamiltonian(parameters)
    #hamiltonian.print(40)
    n=parameters.chain_length**2*parameters.steps
    differencelist=[0 for i in range(0,2*n)]
    old_green_function=[[[1.0+1j for x in range(parameters.chain_length)] for y in range(parameters.chain_length)] for z in range(0,parameters.steps)] 
    difference=100.0
    count=0
    while (difference>0.0001 and count < 10) :
        count+=1

        for r in range(0,parameters.steps):#this initially creates the non-interacting green functions. It then updates using a diagonal self energy.
            gf_int_up[r]=green_function_calculator( hamiltonian ,self_energy_up[r] ,  parameters, energy[r] , r)
            gf_int_down[r]=green_function_calculator( hamiltonian , self_energy_down[r], parameters, energy[r], r)
        #spin_up_occup are included within the self energy as well. Spin_up_occup is only included so we can view there value
        self_energy_up, self_energy_down, spin_up_occup , spin_down_occup =inner_dmft(parameters, gf_int_up, gf_int_down, energy)

        print( "In the ",  count, "first DMFT loop the spin occupation is " , spin_up_occup)
        for r in range(0,parameters.steps):
                for i in range(0,parameters.chain_length):
                    for j in range(0, parameters.chain_length): #this is due to the spin_up_occup being of length chain_length
                    
                        differencelist[r+i+j]=abs(gf_int_up[r][i][j].real-old_green_function[r][i][j].real)
                        differencelist[n+r+i+j]=abs(gf_int_up[r][i][j].imag-old_green_function[r][i][j].imag)
                        old_green_function[r][i][j]=gf_int_up[r][i][j]
                                           
        difference=max(differencelist)
        print("The difference is " , difference, "The count is " , count)
        #print("The mean difference is ", np.mean(differencelist))
        
    for r in range(0,parameters.steps):
        spectral_function_up[r]=spectral_function_calculator(gf_int_up[r], parameters)
        spectral_function_down[r]=spectral_function_calculator(gf_int_down[r], parameters)    
    
    for i in range(0, parameters.chain_length):
        fig = plt.figure()
        
        plt.plot(energy, [e[i][i].imag for e in self_energy_up], color='blue', label='imaginary self energy' ) 
        plt.plot(energy, [e[i][i].real for e in self_energy_up] , color='red' , label='real self energy') 
        plt.title("Many-body self energy")
        plt.legend(loc='upper right')
        plt.xlabel("energy")
        plt.ylabel("Self Energy")  
        plt.show()
    print("The spin up occupaton probability is ", spin_up_occup)
    return gf_int_up, gf_int_down, spectral_function_up, spectral_function_down, spin_up_occup, spin_down_occup


def analytic_gf_1site(parameters):
    analytic_gf= [ 0 for i  in range(parameters.steps)]# this assume the interaction between the scattering region and leads is nearest neighbour 
    energy = [0 for i in range(parameters.steps)]   
    
    f= open(r"C:\Users\user\Desktop\Green function code\Green's Function\embedding_self_energy.txt", "r")
    lines = f.read().split(',')  
    for r in range(0,parameters.steps):  
        energy[r] = float( lines[ 5 * r ] )   
        x= energy[r]-parameters.onsite- float(lines[3+r*5])-float(lines[1+r*5])
        y = (  float(lines[2+r*5]) + float(lines[4+r*5]) ) 
        analytic_gf[r] = x / ( x * x + y * y ) + 1j * y / ( x * x +y * y )
    f.close()
    
    plt.plot(energy, [ e.imag for e in analytic_gf ], color='blue', label='imaginary green function' ) 
    plt.plot(energy, [e.real for e in analytic_gf] , color='red' , label='real green function') 
    plt.title(" Analytical Green function")
    plt.legend(loc='upper right')
    plt.xlabel("energy")
    plt.ylabel("Green Function")  
    plt.show()
    
def analytic_gf_2site(parameters):
    analytic_gf= [ 0 for i  in range(parameters.steps)]# this assume the interaction between the scattering region and leads is nearest neighbour 
    energy = [0 for i in range(parameters.steps)]   
    
    f= open(r"C:\Users\user\Desktop\Green function code\Green's Function\embedding_self_energy.txt", "r")
    lines = f.read().split(',')  
    for r in range(0,parameters.steps):  
        energy[r] = float( lines[ 5 * r ] )   
        x= energy[r]-parameters.onsite- float(lines[3+r*5])
        y = (  - float(lines[2+r*5])  ) 
        a = x * x - y * y - parameters.hopping * parameters.hopping 
        b =2 * x * y
        analytic_gf[r] =  ( a * x + b * y ) / ( a * a + b * b ) + 1j * ( y * a - x * b ) / ( a * a + b * b )
    f.close()
    
    plt.plot(energy, [ e.imag for e in analytic_gf ], color='blue', label='imaginary green function' ) 
    plt.plot(energy, [e.real for e in analytic_gf] , color='red' , label='real green function') 
    plt.title(" Analytical Green function")
    #plt.legend(loc='upper right')
    plt.xlabel("energy")
    plt.ylabel("Green Function")  
    plt.show()

def coupling_matrices(parameters , se_r ):
    coupling_mat = [ create_matrix( parameters.chain_length ) for r in range ( parameters.steps) ]
    for r in range( 0 , parameters.steps ):
        for i in range( parameters.chain_length ):
            for j in range( parameters.chain_length ):                
                coupling_mat[r][i][j] = 1j * ( se_r[r][i][j] - np.conjugate( se_r[r][j][i] ) )
    return coupling_mat

def current_Meir_wingreen(parameters, spectral_function , lesser_gf , left_se_r , right_se_r , energy , v_l , v_r):
    coupling_right = coupling_matrices(parameters, right_se_r)
    coupling_left = coupling_matrices(parameters, left_se_r)

    integrand = [ [ 0 for i in range( parameters.chain_length ) for r in range( parameters.steps ) ] ]
    
    #print(integrand)
    
    for r in range(0 , parameters.steps ):
        for i in range(0 , parameters.chain_length):
            for k in range(0 , parameters.chain_length):
                integrand[i][r]  += ( fermi_function( energy[r] + v_l , parameters) * coupling_left[r][i][k] - fermi_function( energy[r] + v_r , parameters) * coupling_right[r][i][k] ) * spectral_function[r][k][i] + 1j * ( coupling_left[r][i][k] -coupling_right[r][i][k] ) * lesser_gf[r][k][i] 

    trace = [ 0 for r in range(parameters.steps) ]
    
    for r in range(0 , parameters.steps ):
        for i in range(0 , parameters.chain_length):
            trace[r] +=  integrand[i][r]

    current = trace_integrate( energy , parameters , trace ) 
    
    return current


def trace_integrate(energy, parameters, trace):
    current = 0
    delta_energy = (parameters.e_upper_bound-parameters.e_lower_bound)/parameters.steps
    
    for r in range(0,parameters.steps):                
        current += delta_energy * trace[r]
    return current
       




    
def main():
    time_start = time.perf_counter()
    onsite, gamma, hopping, chemical_potential, temperature , hubbard_interaction = 1.0 , 2.0 , -1.0 ,0.0, 0.0 , 0.3
    chain_length=1
    steps=81 #number of energy points
    e_upper_bound , e_lower_bound = 10.0 , -10.0
    
    spin_up_occup , spin_down_occup = [ 0.0 for x in range(0, chain_length)] , [ 1.0 for x in range(0, chain_length)]

    parameters=Parameters(onsite, gamma, hopping, chain_length, chemical_potential, temperature, steps, e_upper_bound ,e_lower_bound, hubbard_interaction)

    v_l=0
    v_r=0

    energy=[e_lower_bound+(e_upper_bound-e_lower_bound)/steps*x for x in range(steps)]
    green_function_up=[create_matrix(chain_length) for z in range(0,steps)] 
    green_function_down=[create_matrix(chain_length) for z in range(0,steps)]
    
    spectral_function_up=[create_matrix(chain_length) for z in range(0,steps)] 
    spectral_function_down=[create_matrix(chain_length) for z in range(0,steps)]
    #this creates [ [ [0,0,0] , [0,0,0],, [0,0,0] ] , [0,0,0] , [0,0,0],, [0,0,0] ] ... ], ie one chain_length by chain_length 
    # dimesional create_matrix for each energy. The first index in spectral function refers to what energy we are selcting. 
    #the next two indices refer to which enter in our create_matrix we are selecting.    
    green_function_up, green_function_down, spectral_function_up, spectral_function_down, spin_up_occup, spin_down_occup = gf_dmft(parameters,  energy )

    magnetisation=[spin_up_occup[i]-spin_down_occup[i] for i in range(0,chain_length)]
    #analytic2=[(2/np.pi)*gamma/((energy[x]-onsite-hubbard_interaction*spin_up_occup[-1]+hubbard_interaction*spin_down_occup[-1]*spin_up_occup[-1])**2+4*gamma**2) for x in range(steps)]   
    print("The magnetisation is ", magnetisation)
    #print(count)
    
    fig = plt.figure()
   
    """
    for i in range(0,chain_length):
        plt.plot(energy, [ e[i][i] for e in spectral_function_up]  , color='blue' , label='Spectral up' ) 
        plt.plot(energy, [ -e[i][i] for e in spectral_function_down], color='red', label='Spectral down')
        #plt.plot(energy, dos_spin_up[i] , color='blue', label='spin up DOS' ) 
        #plt.plot(energy, dos_spin_down[i], color='red', label='spin down DOS')
#   
   # plt.plot(energy,analytic, color='tomato')
    #plt.plot(energy,analytic2, color='green')
    plt.title("Converged Spectral function")
    plt.legend(loc='upper right')
    plt.xlabel("energy")
    plt.ylabel("Sepctral Function")  
    plt.show()
    """
    for i in range(0,chain_length):

        plt.plot(energy, [ e[i][i].real for e in green_function_up]  , color='red' , label='Real Green up' ) 
        plt.plot(energy, [ e[i][i].imag for e in green_function_up], color='blue', label='Imaginary Green function')
        #plt.plot(energy, dos_spin_up[i] , color='blue', label='spin up DOS' ) 
        #plt.plot(energy, dos_spin_down[i], color='red', label='spin down DOS')
#   
   # plt.plot(energy,analytic, color='tomato')
    #plt.plot(energy,analytic2, color='green')
    plt.title("Converged Green function")
    #plt.legend(loc='upper right')
    plt.xlabel("energy")
    plt.ylabel("Green Function")  
    plt.show()  
    
    self_energy_left=[create_matrix(chain_length) for r in range(parameters.steps)]
    self_energy_right=[create_matrix(chain_length)for r in range(parameters.steps)]
        
    f= open(r"C:\Users\user\Desktop\Green function code\Green's Function\embedding_self_energy.txt", "r")
    lines = f.read().split(',')  
    for r in range(0,parameters.steps):  
            self_energy_right[r][-1][-1]=float(lines[3+r*5])+1j*float(lines[4+r*5])
            self_energy_left[r][0][0]=float(lines[1+r*5])+1j*float(lines[2+r*5])  
    f.close()
    

    lesser_gf=[create_matrix(parameters.chain_length) for z in range(0,parameters.steps)]  
    for r in range(0, parameters.steps):
        for i in range(0, parameters.chain_length):
            for j in range(0, parameters.chain_length):
                lesser_gf[r][i][j]= -2j*fermi_function( energy[r], parameters)*(green_function_up[r][i][j]-np.conjugate(green_function_up[r][j][i]))  

    
    current = current_Meir_wingreen( parameters , spectral_function_up , lesser_gf, self_energy_left , self_energy_right , energy, v_l, v_r)
    
    print("The current is " , current )
    analytic_gf_1site(parameters)
    time_elapsed = (time.perf_counter() - time_start)
    print(" The time it took the computation is" , time_elapsed)
            
if __name__=="__main__":#this will only run if it is a script and not a import module
    main()

