#from mpmath import mpc , mp
import scipy.linalg as la
import matplotlib.pyplot as plt
import math
import time
import leads_self_energy
import parameters
import warnings
from typing import List

class HubbardHamiltonian: #this is the hamiltonian of the scattering region
    matrix: None
    effective_hamiltonian=None
    self_energy_left=None
    self_energy_right=None
   
    def __init__(self ):
        self.matrix=create_matrix( parameters.chain_length )
        self.effective_hamiltonian=[create_matrix( parameters.chain_length ) for r in range(parameters.steps )]
        self.self_energy_left=[create_matrix(parameters.chain_length) for r in range(parameters.steps )]
        self.self_energy_right=[create_matrix(parameters.chain_length)for r in range(parameters.steps )]
        #this willgetting the embedding slef energies from the leads code        
        self.self_energy_left , self.self_energy_right = self.embedding_self_energy()
        #this defines the hamiltonian and effective hamiltonian.       
        for r in range(0,parameters.steps):
            for i in range(0,parameters.chain_length-1):
                self.matrix[i][i+1]=parameters.hopping
                self.matrix[i+1][i]=parameters.hopping
            for i in range(0,parameters.chain_length):
                self.matrix[i][i]=parameters.onsite
                for j in range(0,parameters.chain_length):  
                    self.effective_hamiltonian[r][i][j]=self.matrix[i][j] + self.self_energy_right[r][i][j] + self.self_energy_left[r][i][j]

    def embedding_self_energy(self ):#this function gets the retarded embedding self energies from the leads codes
        se_emb_l=[create_matrix(parameters.chain_length) for r in range(parameters.steps)]
        se_emb_r=[create_matrix(parameters.chain_length)for r in range(parameters.steps)]
        
        f= open(r"C:\Users\user\Desktop\Green function code\Green's Function\embedding_self_energy.txt", "r")
        lines = f.read().split(',')  
        for r in range(0,parameters.steps ):  
            se_emb_r[r][-1][-1]=float(lines[3+r*5])+1j*float(lines[4+r*5])
            se_emb_l[r][0][0]=float(lines[1+r*5])+1j*float(lines[2+r*5])  
        f.close()
        
        return se_emb_l , se_emb_r
    
    def plot_embedding_self_energy( self ):#this allows me to plot the embedding self energies. Not called in this code.
            plt.plot( parameters.energy , [ e[-1][-1].real for e in self.self_energy_right]  , color='blue', label='real self energy' ) 
            plt.plot( parameters.energy , [ e[-1][-1].imag for e in self.self_energy_right], color='red', label='imaginary self energy')
            plt.title("embedding self energy")
   # plt.plot(energy,analytic, color='tomato')
    #plt.plot(energy,analytic2, color='green')
            plt.legend(loc='upper right')
            plt.xlabel("energy")
            plt.ylabel("embedding self energy")  
            plt.show()
            
    def print_hamiltonian(self, num: int):# this allows me to print the effective hamiltonian if called for a certain energy point specified by num. 
    # eg. hamiltonian.print(4) will print the effective hamiltonian of the 4th energy step
        for i in range(0,parameters.chain_length):
            row_string = " ".join((str(r).rjust(5, " ") for r in self.efffective_matrix[num][i])) #rjust adds padding, join connects them all
            print(row_string)


def fermi_function( energy: complex ):
    if( parameters.temperature == 0 ):
        if( energy.real < parameters.chemical_potential ):
            return 1
        else:
            return 0
    else:
        return 1 / (1 + math.exp( ( energy.real - parameters.chemical_potential ) / parameters.temperature ))

"""
#this code allows me to increase the numerical precision in the integrating function
def integrate(  raw_gf_1, raw_gf_2, raw_gf_3, r):# in this function, the green functions are 1d arrays in energy. this is becasue we have passed the diagonal component of the green fun( lesser, or retarded) 
    delta_energy = ( parameters.e_upper_bound - parameters.e_lower_bound ) / parameters.steps
    result = 0    
    mp.dps = 30
    gf_1 = [ mpc(e) for e in raw_gf_1] 
    gf_2 = [ mpc(e) for e in raw_gf_2] 
    gf_3 = [ mpc(e) for e in raw_gf_3]     
    for i in range(0,parameters.steps ):
        for j in range(0,parameters.steps ):
            #print("i = " , i , " j = ", j , " r = ", r)
            if ( ( (i+j-r) >= 0 ) and ( (i+j-r) < parameters.steps ) ):
                #if( r == 0):
                    #energy = parameters.e_lower_bound +  delta_energy * (i + j - r) 
                    #print("the energy is ", energy , i , j)
                    
                result=(delta_energy/(2*parameters.pi))**2 * gf_1[i] * gf_2[j] * gf_3[ (i+j-r) ] +result
                #print("the energy is ", energy , i , j)                
    return complex(result)
"""

def integrate(  gf_1: List[complex], gf_2: List[complex], gf_3: List[complex], r: int):
    # in this function, the green functions are 1d arrays in energy. this is becasue we have passed the diagonal component of the green fun( lesser, or retarded).The  
    delta_energy = ( parameters.e_upper_bound - parameters.e_lower_bound ) / parameters.steps
    result = 0    
    for i in range(0,parameters.steps ):
        for j in range(0,parameters.steps ):
            if ( ( (i+j-r) >= 0 ) and ( (i+j-r) < parameters.steps ) ):
            #this integrates like PHYSICAL REVIEW B 74, 155125 2006
            # I say the green function is zero outside -14 and +14. This means I need the final green function in the integral to be within an energy of -14 
            #and 14. The index of 0 corresponds to -14. Hence we need i+J-r>0 but in order to be less an energy of 14 we need i+j-r<steps. These conditions enesure the enrgy of the gf3 greens function to be within (-14, 14)
            
                result = (delta_energy/(2*parameters.pi))**2 * gf_1[i] * gf_2[j] * gf_3[ i+j-r ] +result
            else:
                result = result
            
    return result

def green_lesser_fluctuation_dissiption(green_function: List[List[List[complex]]]): #this is only used to compare the lesser green functions using two different methods. This is not used in the calculation of the self energies.
    g_lesser = [ create_matrix(1) for z in range( 0 , parameters.steps )]  
    for r in range( 0 , parameters.steps ):
        g_lesser[r][0][0] = - fermi_function( parameters.energy[r])*(green_function[r][0][0]-conjugate(green_function[r][0][0]))  
    return g_lesser

def self_energy_calculator( g_0_up: List[List[List[complex]]], g_0_down: List[List[List[complex]]]  , gf_lesser_up: List[List[List[complex]]], gf_lesser_down: List[List[List[complex]]]):# this creates the entire parameters.energy() array at once
    self_energy=[create_matrix(parameters.chain_length) for z in range(0,parameters.steps)]   

    for r in range(0,parameters.steps):# the are calculating the self parameters.energy() sigma_{ii}(E) for each discretized parameters.energy(). To do this we pass the green_fun_{ii} for all energies as we need to integrate over all energies in the integrate function
        for i in range(0, 1):
            self_energy[r][i][i] = parameters.hubbard_interaction**2*( integrate(  [ e[i][i] for e in g_0_up] , [ e[i][i] for e in g_0_down]  , [ e[i][i] for e in gf_lesser_down]   , r )  )
            self_energy[r][i][i] += parameters.hubbard_interaction**2*( integrate( [ e[i][i] for e in g_0_up] , [ e[i][i] for e in gf_lesser_down] , [ e[i][i] for e in gf_lesser_down]  ,r  ) ) 
            self_energy[r][i][i] += parameters.hubbard_interaction**2*( integrate( [ e[i][i] for e in gf_lesser_up] , [ e[i][i] for e in g_0_down] , [ e[i][i] for e in gf_lesser_down]  ,r  ) ) 
            self_energy[r][i][i] += parameters.hubbard_interaction**2*( integrate( [ e[i][i] for e in gf_lesser_up] , [ e[i][i] for e in gf_lesser_down]  , [conjugate( e[i][i]) for e in g_0_down]  ,r  ) ) #fix advanced green function

    return self_energy

def lesser_embedding():# in this function, the green functions are 1d arrays in energy. this is becasue we have passed the diagonal component of the green fun( lesser, or retarded) 
    se_emb_l_lesser = [create_matrix( parameters.chain_length ) for r in range(parameters.steps ) ]
    se_emb_r_lesser = [create_matrix( parameters.chain_length )for r in range(parameters.steps ) ]
        
    f = open(r"C:\Users\user\Desktop\Green function code\Green's Function\embedding_self_energy_lesser.txt", "r")
    lines = f.read().split(',')  
    for r in range(0,parameters.steps ):  
        se_emb_r_lesser[r][-1][-1] = float(lines[2 + r * 4]) + 1j * float( lines[3 + r * 4] )
        se_emb_l_lesser[r][0][0] = float(lines[ r * 4 ])+1j*float(lines[ 1 + r * 4])  
    f.close()
        
    return se_emb_l_lesser , se_emb_r_lesser

def create_matrix( size: int ):
    return [ [ 0.0 for x in range( size ) ] for y in range( size )]
          
def green_function_calculator( hamiltonian , self_energy: List[List[complex]]  ,  energy: float, energy_step: int):#this calculates the retarded green function for a specific energy point.
#the embedding self energies are within the effective hamiltonian and the self energy array is the many body self energy.
    inverse_green_function=create_matrix(parameters.chain_length)
    for i in range( 0 , parameters.chain_length ):
           for j in range( 0 , parameters.chain_length ): 
               if( i == j ):                
                   inverse_green_function[i][j] = - hamiltonian.effective_hamiltonian[energy_step][i][j] - self_energy[i][i] + energy
               else:
                   inverse_green_function[i][j] = - hamiltonian.effective_hamiltonian[energy_step][i][j]
        #print(inverse_green_function[i])
    #hamiltonian.print()
    
    return la.inv(inverse_green_function , overwrite_a=False , check_finite=True )    


def spectral_function_calculator( green_function: List[List[complex]]  ):
    spectral_function = create_matrix( parameters.chain_length )
    for i in range( 0 , parameters.chain_length ):
        for j in range( 0 , parameters.chain_length ):
            spectral_function[i][j] = 1j * (green_function[i][j] - conjugate(green_function[j][i]))  
    return spectral_function    

def conjugate(x):
    a = x.real
    b = x.imag
    y = a - 1j * b
    return y


def get_spin_occupation( gf_lesser_up: List[complex] , gf_lesser_down: List[complex]  ):#this should work as in first order interaction, it gives the same result as fluctuation dissaption thm to 11 deciaml places
    delta_energy = (parameters.e_upper_bound - parameters.e_lower_bound )/parameters.steps
    result_up , result_down = 0 , 0 
    for r in range( 0 , parameters.steps ):
        result_up = (delta_energy) * gf_lesser_up[r] + result_up
        result_down = (delta_energy) * gf_lesser_down[r] + result_down
    x= -1j / (parameters.pi) * result_up 
    y= -1j / (parameters.pi) * result_down     
    return x , y

def gf_lesser_nq( gf: List[List[List[complex]]] ,  se_mb_lesser: List[List[List[complex]]] ): #this obtains the lesser green function. The se_mb_lesser is the second order lesser many body self energy.
    gf_lesser = [ create_matrix( parameters.chain_length ) for z in range( 0 , parameters.steps ) ]     
    self_energy_left_lesser  , self_energy_right_lesser = lesser_embedding()

    for r in range( 0 , parameters.steps ):
        for i in range( 0 , parameters.chain_length ):
            for j in range( 0 , parameters.chain_length ):    
                for k in range(0 , parameters.chain_length ):
                   #this assumes that the self energy is diagonal
                      gf_lesser[r][i][j] += gf[r][i][k] * ( self_energy_left_lesser[r][k][k] + self_energy_right_lesser[r][k][k] + se_mb_lesser[r][k][k]  ) * conjugate( gf[r][j][k] ) #this additionaly number prevents the cases where the SE is initially zero and as a result everything is always zero

    return gf_lesser

def lesser_se_mb( gf_r_down: List[List[List[complex]]] , gf_lesser_down: List[List[List[complex]]] , gf_lesser_up: List[List[List[complex]]] ):#this code obtains the second order lesser many body self energy.
    self_energy_up_lesser = [ create_matrix(parameters.chain_length) for i in range( parameters.steps ) ]   
    gf_a_down = [ create_matrix(parameters.chain_length) for i in range( parameters.steps ) ]
    gf_greater_down = [ create_matrix(parameters.chain_length) for i in range( parameters.steps ) ]   
    for r in range(0 , parameters.steps ):
        for i in range(0, parameters.chain_length ):
            for j in range(0, parameters.chain_length ):
                gf_a_down[r][i][j] = conjugate(gf_r_down[r][j][i])
                gf_greater_down[r][i][j] = gf_r_down[r][i][j] - gf_a_down[r][i][j] + gf_lesser_down[r][i][j]  
    for r in range( 0 , parameters.steps):
        for i in range(0 , parameters.chain_length):
            self_energy_up_lesser[r][i][i] = parameters.hubbard_interaction**2 * integrate( [ e[i][i] for e in gf_lesser_up ] , [ e[i][i] for e in gf_lesser_down ]  , [ e[i][i] for e in gf_greater_down ]   , r )  
    return self_energy_up_lesser

"""   
def lesser_se_mb_eq( gf_r_up , gf_r_down ):
    self_energy_up_lesser = [ create_matrix(parameters.chain_length) for i in range( parameters.steps ) ]   
    gf_lesser_down = [ create_matrix(parameters.chain_length) for i in range( parameters.steps ) ]
    gf_lesser_up = [ create_matrix(parameters.chain_length) for i in range( parameters.steps ) ]
    gf_greater_down = [ create_matrix(parameters.chain_length) for i in range( parameters.steps ) ]   
    for r in range(0 , parameters.steps ):
        for i in range(0, parameters.chain_length ):
            for j in range(0, parameters.chain_length ):
                gf_greater_down[r][i][j] = ( 1.0 - fermi_function(parameters.energy[r].real ) ) * (gf_r_down[r][i][j]  - conjugate(gf_r_down[r][j][i]))
                gf_lesser_down[r][i][j] = -fermi_function(parameters.energy[r].real )  * (gf_r_down[r][i][j]  - conjugate(gf_r_down[r][j][i]))
                gf_lesser_up[r][i][j] = -fermi_function(parameters.energy[r].real )  * (gf_r_up[r][i][j]  - conjugate(gf_r_up[r][j][i]))
                #gf_greater_down[r][i][j] = (1.0 - fermi_function(parameters.energy[r].real)) * (gf_r_down[r][j][i] - gf_a_down[r][j][i])
    
    warnings.warn('Dear future Declan,  Please change this for when you do for than 1 orbital in the scattering region. Your sincerely, past Declan ')
    for r in range( 0 , parameters.steps):
        self_energy_up_lesser[r][0][0] = parameters.hubbard_interaction**2 * integrate( [ e[0][0] for e in gf_lesser_up ] , [ e[0][0] for e in gf_lesser_down ]  , [ e[0][0] for e in gf_greater_down ]   , r ) 
        
    return self_energy_up_lesser 
"""

def inner_dmft( gf_int_up: List[List[List[complex]]] , gf_int_down: List[List[List[complex]]] , gf_int_lesser_up: List[List[List[complex]]] , gf_int_lesser_down: List[List[List[complex]]] ): #this can solve the impurity problem self consistently. Currently it only returns the many body self energy, the retarded and lesser for spin up and down.
    g_local_up = [ create_matrix(1) for i in range( parameters.steps ) ]#This function can be made more efficient but I belive does the job for now.
    g_local_down = [ create_matrix(1) for i in range( parameters.steps ) ]
    self_energy_up = [ create_matrix( parameters.chain_length ) for z in range( 0 , parameters.steps ) ]  
    self_energy_down = [ create_matrix( parameters.chain_length ) for z in range( 0 , parameters.steps ) ] 
    self_energy_up_lesser = [ create_matrix( parameters.chain_length ) for z in range( 0 , parameters.steps ) ] 
    self_energy_down_lesser = [create_matrix( parameters.chain_length ) for z in range( 0 , parameters.steps  ) ] 

    local_sigma_up , local_sigma_down = [ create_matrix(1) for i in range( parameters.steps )] , [create_matrix(1) for i in range(parameters.steps)]
    spin_up_occup , spin_down_occup = [ 0.0 for x in range(0, parameters.chain_length )] , [ 0.0 for x in range(0, parameters.chain_length)]
    local_sigma_down = [ create_matrix(1) for i in range( parameters.steps )]

    """ #this is for when we want to get the anderson impurity self consistently
    hamiltonian = HubbardHamiltonian()
    g_initial = [ create_matrix( parameters.chain_length ) for z in range( 0 , parameters.steps ) ] 
    
    for r in range( 0 , parameters.steps ):
        g_initial[r] = green_function_calculator( hamiltonian , self_energy_up[r] ,  parameters.energy[r] , r )
    old_green_function = [0 for z in range( 0 , parameters.steps )] 
    n = parameters.chain_length * parameters.steps       
    differencelist = [0 for i in range( 0, 2 * n ) ]    
    """
    difference = 100.0
    for i in range( 0 , parameters.chain_length ):         
        for r in range( 0 , parameters.steps ):#this sets the impurity green function to the local lattice green function for each lattice site(the i for loop)
            g_local_up[r][0][0] = gf_int_up[r][i][i]
            g_local_down[r][0][0] = gf_int_down[r][i][i]
                                            
        while( difference > 0.0001 ):#this is solving the impurity problem self consistently which in principle should be correct
            local_spin_up , local_spin_down = get_spin_occupation( [ e[0][0] for e in gf_int_lesser_up ] ,  [ e[0][0] for e in gf_int_lesser_down ] )
            local_sigma_up = self_energy_calculator( g_local_up , g_local_down ,  gf_int_lesser_up , gf_int_lesser_down )
            local_sigma_down = self_energy_calculator( g_local_down , g_local_up , gf_int_lesser_down , gf_int_lesser_up )
            
            for r in range( 0 , parameters.steps ):

                 # to make first order u should remove the += and just have a = sign
                 #local_sigma_up[r][0][0] = parameters.hubbard_interaction * local_spin_down
                 #local_sigma_down[r][0][0] = parameters.hubbard_interaction * local_spin_up                
                 # to make first order u should remove the += and just have a = sign
                 local_sigma_up[r][0][0] += parameters.hubbard_interaction * local_spin_down
                 local_sigma_down[r][0][0] += parameters.hubbard_interaction * local_spin_up
                 """#this is for when we want to get the anderson impurity self consistently
                 g_initial_up[r] = 1 / ( ( 1 / g_local_up[r][0][0]) + local_sigma_up[r][0][0] )# this is getting the new dynamical mean field
                 g_initial_down[r] = 1 / ( ( 1 / g_local_down[r][0][0]) + local_sigma_down[r][0][0] )
                 
                 g_local_up[r][0][0]=1/((1/g_initial_up[r])-local_sigma_up[r][0][0])
                 g_local_down[r][0][0]=1/((1/g_initial_down[r])-local_sigma_down[r][0][0])  

            for r in range(0,parameters.steps ):
                        differencelist[r]=abs(g_local_up[r][0][0].real-old_green_function[r].real)
                        differencelist[n+r]=abs(g_local_up[r][0][0].imag-old_green_function[r].imag)
                        old_green_function[r]=g_local_up[r][0][0]
            """
            #difference=max(differencelist)
            difference=0

        for r in range( 0 , parameters.steps ): #this then returns a diagonal self energy
            self_energy_up[r][i][i] = local_sigma_up[r][0][0]
            self_energy_down[r][i][i] = local_sigma_down[r][0][0]   
        spin_up_occup[i] = local_spin_up
        spin_down_occup[i] = local_spin_down

    self_energy_up_lesser = lesser_se_mb( gf_int_down , gf_int_lesser_down , gf_int_lesser_up )
    self_energy_down_lesser = lesser_se_mb( gf_int_up , gf_int_lesser_up , gf_int_lesser_down )
    
    return self_energy_up , self_energy_down , self_energy_up_lesser , self_energy_down_lesser , spin_up_occup , spin_down_occup


def gf_dmft(voltage: int): # this function gets the converged green function using dmft to get the self energy.
    gf_int_up = [ create_matrix( parameters.chain_length ) for z in range( 0 , parameters.steps ) ] 
    gf_int_down = [ create_matrix( parameters.chain_length ) for z in range( 0 , parameters.steps ) ]
    spectral_function_up = [ create_matrix( parameters.chain_length ) for z in range( 0 , parameters.steps ) ] 
    spectral_function_down = [ create_matrix( parameters.chain_length ) for z in range( 0 , parameters.steps ) ]

    se_mb_up = [ create_matrix( parameters.chain_length ) for z in range( 0 , parameters.steps )] 
    se_mb_down = [ create_matrix( parameters.chain_length ) for z in range( 0 , parameters.steps )] 
    se_mb_up_lesser = [ create_matrix( parameters.chain_length ) for z in range( 0 , parameters.steps )] 
    se_mb_down_lesser = [ create_matrix( parameters.chain_length ) for z in range( 0 , parameters.steps )] #these are the same for spin up and spin down
    
    hamiltonian = HubbardHamiltonian() # this sets up the effective hamiltonian

    n = parameters.chain_length**2 * parameters.steps
    differencelist = [0 for i in range(0, 2 * n)]
    old_green_function = [ [ [ 1.0 + 1j for x in range( parameters.chain_length ) ] for y in range( parameters.chain_length ) ] for z in range( 0 , parameters.steps )] 
    difference = 100.0
    count = 0
    while ( difference > 0.0001 and count < 15) : #these allows us to determine self consistency in the retarded green function
        count += 1

        for r in range( 0 , parameters.steps ):#this initially creates the non-interacting green functions. It then updates using a diagonal self energy.
            gf_int_up[r] = green_function_calculator( hamiltonian ,se_mb_up[r] ,  parameters.energy[r] , r)
            gf_int_down[r] = green_function_calculator( hamiltonian , se_mb_down[r],  parameters.energy[r], r) #should be some indexes here
        
        gf_int_lesser_up = gf_lesser_nq( gf_int_up , se_mb_up_lesser )
        gf_int_lesser_down = gf_lesser_nq(  gf_int_down , se_mb_down_lesser )            

        #spin_up_occup are included within the self energy as well. Spin_up_occup is only included so we can view there value. These self energies are diagonal. We will use them again to obtain a new greens function. Repeat until self consistent.
        se_mb_up , se_mb_down , se_mb_up_lesser , se_mb_down_lesser , spin_up_occup , spin_down_occup = inner_dmft( gf_int_up , gf_int_down , gf_int_lesser_up , gf_int_lesser_down )

        print( "In the ",  count, "first DMFT loop the spin occupation is " , spin_up_occup)
        for r in range( 0 , parameters.steps ):
                for i in range( 0 , parameters.chain_length ):
                    for j in range( 0 , parameters.chain_length ): #this is due to the spin_up_occup being of length chain_length
                    
                        differencelist[ r + i + j ] = abs( gf_int_up[r][i][j].real - old_green_function[r][i][j].real )
                        differencelist[n + r + i + j] = abs( gf_int_up[r][i][j].imag - old_green_function[r][i][j].imag )
                        old_green_function[r][i][j] = gf_int_up[r][i][j]
                                           
        difference = max(differencelist)
        #print("The difference is " , difference, "The count is " , count)
        #print(" ")
        #print("The mean difference is ", np.mean(differencelist))
        if ( parameters.hubbard_interaction == 0):
            break
        
    #once converged we get the spectral function and we plot the many body self energy
    for r in range( 0 , parameters.steps ):
        spectral_function_up[r] = spectral_function_calculator(gf_int_up[r])
        spectral_function_down[r] = spectral_function_calculator(gf_int_down[r])    
    """
    for i in range( 0, parameters.chain_length ):
        fig = plt.figure()
        
        plt.plot(parameters.energy , [e[i][i].imag for e in se_mb_up], color='blue', label='imaginary self energy' ) 
        plt.plot(parameters.energy , [e[i][i].real for e in se_mb_up] , color='red' , label='real self energy') 
        plt.title("Many-body self energy")
        plt.legend(loc='upper right')
        plt.xlabel("energy")
        plt.ylabel("Self Energy")  
        plt.show()
    """
    print("The spin up occupaton probability is ", spin_up_occup)
    if(voltage == 0):#this compares the two methods in equilibrium
        compare_g_lesser(gf_int_lesser_up , gf_int_up)
        print("happened once")
        
    return gf_int_up, gf_int_down, spectral_function_up, spectral_function_down, spin_up_occup, spin_down_occup , gf_int_lesser_up 


def compare_g_lesser( g_lesser_up: List[List[List[complex]]], gf_int_up: List[List[List[complex]]]):# this function compare the lesser green function in equilibrium from the two methods.
    lesser_g = green_lesser_fluctuation_dissiption( gf_int_up )
    
    difference = -1000
    count = 0
    for r in range(0 , parameters.steps ):
        for i in range(0 , parameters.chain_length ):
            for j in range(0 , parameters.chain_length ):
                if( abs(g_lesser_up[r][i][j].real - lesser_g[r][i][j].real )  > difference ):
                    difference = abs(g_lesser_up[r][i][j].real - lesser_g[r][i][j].real )
                    count = r
                if( abs(g_lesser_up[r][i][j].imag - lesser_g[r][i][j].imag )  > difference ):
                    difference = abs(g_lesser_up[r][i][j].imag - lesser_g[r][i][j].imag )  
                    count = r
    print(" The difference between the two methods in the lesser gf is " , difference , ". This occured for count = " , count )
    fig = plt.figure()                
    plt.plot(parameters.energy , [e[0][0].imag for e in g_lesser_up] , color = 'blue'  , label='other imag')
    #plt.plot(parameters.energy , [e[0][0].real for e in g_lesser_up] , color = 'orange'  , label='other real')
    plt.plot(parameters.energy , [e[0][0].imag for e in lesser_g] , color = 'green'  , label='FD imag')
    #plt.plot(parameters.energy , [e[0][0].real for e in lesser_g] , color = 'green'  , label='FD real')
    plt.title(" Numerical GF lesser")
    plt.legend(loc='upper right')
    plt.xlabel("energy")
    plt.ylabel
    
def analytic_gf_1site():#this the analytic soltuion for the noninteracting green function when we have a single site in the scattering region
    analytic_gf = [ 0 for i  in range( parameters.steps ) ]# this assume the interaction between the scattering region and leads is nearest neighbour 
    energy = [ 0 for i in range( parameters.steps ) ]   
    
    f = open(r"C:\Users\user\Desktop\Green function code\Green's Function\embedding_self_energy.txt", "r")
    lines = f.read().split(',')  
    for r in range( 0 , parameters.steps ):  
        energy[r] = float( lines[ 5 * r ] )   
        x = energy[r] - parameters.onsite - float( lines[3 + r * 5] )-float( lines[1 +  r * 5])
        y = (  float(lines[2 + r * 5]) + float(lines[4 + r * 5 ]) ) 
        analytic_gf[r] = x / ( x * x + y * y ) + 1j * y / ( x * x +y * y )
    f.close()

    plt.plot( parameters.energy , [ e.imag for e in analytic_gf ], color='blue', label='imaginary green function' ) 
    plt.plot( parameters.energy , [e.real for e in analytic_gf] , color='red' , label='real green function') 
    plt.title(" Analytical Green function")
    plt.legend(loc='upper right')
    plt.xlabel("energy")
    plt.ylabel("Green Function")  
    plt.show()

def analytic_gf_2site():#this the analytic soltuion for the noninteracting green function when we have 2 sites in the scattering region
    analytic_gf= [ 0 for i  in range( parameters.steps ) ]# this assume the interaction between the scattering region and leads is nearest neighbour 
    energy = [ 0 for i in range( parameters.steps ) ]   
    
    f = open(r"C:\Users\user\Desktop\Green function code\Green's Function\embedding_self_energy.txt", "r")
    lines = f.read().split(',')  
    for r in range( 0 , parameters.steps ):  
        energy[r] = float( lines[ 5 * r ] )   
        x= energy[r] - parameters.onsite - float(lines[3 + r * 5])
        y = (  - float(lines[ 2 + r * 5])  ) 
        a = x * x - y * y - parameters.hopping * parameters.hopping 
        b = 2 * x * y
        analytic_gf[r] =  ( a * x + b * y ) / ( a * a + b * b ) + 1j * ( y * a - x * b ) / ( a * a + b * b )
    f.close()
    
    plt.plot( energy , [ e.imag for e in analytic_gf ], color='blue', label='imaginary green function' ) 
    plt.plot( energy , [e.real for e in analytic_gf] , color='red' , label='real green function') 
    plt.title(" Analytical Green function")
    #plt.legend(loc='upper right')
    plt.xlabel("energy")
    plt.ylabel("Green Function")  
    plt.show()

def coupling_matrices(se_r: List[List[List[complex]]]):# coupling matirces for the current calculation.
    coupling_mat = [ create_matrix( parameters.chain_length ) for r in range ( parameters.steps ) ]
    for r in range( 0 , parameters.steps ):
        for i in range( parameters.chain_length ):
            for j in range( parameters.chain_length ):                
                coupling_mat[r][i][j] = 1j * ( se_r[r][i][j] - conjugate( se_r[r][j][i] ) )
    return coupling_mat

def analytic_current_Meir_wingreen(  voltage_step: int ):#this uses the analytic green function to get the Meir wingreen current. this is not currently called
    left_se_r = [create_matrix( parameters.chain_length ) for r in range(parameters.steps)]
    right_se_r = [create_matrix( parameters.chain_length )for r in range(parameters.steps)]

    f= open(r"C:\Users\user\Desktop\Green function code\Green's Function\embedding_self_energy.txt", "r")
    lines = f.read().split(',')  
    for r in range(0, parameters.steps ):  
            right_se_r[r][-1][-1] = float(lines[3+r*5])+1j*float(lines[4+r*5])
            left_se_r[r][0][0] = float(lines[1+r*5])+1j*float(lines[2+r*5])  
    f.close()
        
    self_energy_left_lesser , self_energy_right_lesser = lesser_embedding()
    analytic_gf =     analytic_gf_1site()
    analytical_g_lesser = [0 for r in range(parameters.steps)]   

    for r in range(0 , parameters.steps):
        analytical_g_lesser[r] = analytic_gf[r] * ( self_energy_left_lesser [r][0][0] +self_energy_right_lesser[r][0][0] ) * conjugate(analytic_gf[r])  
   
    analytical_spectral = [ 1j * ( analytic_gf[r] - conjugate(analytic_gf[r]) ) for r in range(parameters.steps) ]
    
    coupling_right = coupling_matrices( right_se_r)
    coupling_left = coupling_matrices( left_se_r)

    trace = [ 0 for r in range(parameters.steps ) ]
    for r in range(0 , parameters.steps ):
        trace[r] = -(fermi_function(parameters.energy[r] - parameters.voltage_l[voltage_step] ) * coupling_left[r][0][0] - fermi_function(parameters.energy[r] - parameters.voltage_r[voltage_step] ) * coupling_right[r][0][0] ) * analytical_spectral[r] + 1j * ( coupling_left[r][0][0] - coupling_right[r][0][0]) * analytical_g_lesser[r]
    

    
    current = trace_integrate(trace) 
    return current

def current_Meir_wingreen( spectral_function: List[List[List[complex]]] , lesser_gf: List[List[List[complex]]] , left_se_r: List[List[List[complex]]] , right_se_r: List[List[List[complex]]] , voltage_step: int ):
    coupling_right = coupling_matrices( right_se_r)
    coupling_left = coupling_matrices( left_se_r)
    
    integrand = [ [ 0 for i in range( parameters.chain_length ) for r in range( parameters.steps ) ] ]

    trace = [ 0 for r in range(parameters.steps ) ]
    warnings.warn('Dear future Declan,  This assumes that the gf is the same for spin up and down. Your sincerely, past Declan ')

    for r in range(0 , parameters.steps ):
        for i in range(0 , parameters.chain_length ):
            for k in range(0 , parameters.chain_length ):#factor of two comes from the spin. This cancels with a factor of two in the formula. This is from the paper. PHYSICAL REVIEW B 72, 125114 2005
                integrand[i][r]  =  ( ( fermi_function( - parameters.voltage_l[voltage_step] + parameters.energy[r].real ) * coupling_left[r][i][k] - fermi_function( - parameters.voltage_r[voltage_step]  + parameters.energy[r].real ) * coupling_right[r][i][k] ) * spectral_function[r][k][i] + 1j * ( coupling_left[r][i][k] - coupling_right[r][i][k] ) * lesser_gf[r][k][i] )
    
    
    for r in range(0 , parameters.steps  ):
        for i in range(0 , parameters.chain_length ):
            trace[r] +=  integrand[i][r]
    current = trace_integrate(trace) 
    return current


def trace_integrate( trace: List[float]):
    current = 0
    delta_energy = ( parameters.e_upper_bound - parameters.e_lower_bound ) / parameters.steps
    
    for r in range( 0 , parameters.steps ):                
        current += delta_energy * trace[r] / (parameters.pi * 2)
    return current
        
def landauer_current( gf_r: List[List[List[complex]]] , left_se_r: List[List[List[complex]]] , right_se_r: List[List[List[complex]]] , voltage_step: int ):
    coupling_right = coupling_matrices( right_se_r)
    coupling_left = coupling_matrices( left_se_r)
    
    gf_a = [ create_matrix(parameters.chain_length) for i in range( parameters.steps ) ]  
    for r in range(0 , parameters.steps ):
        for i in range(0, parameters.chain_length ):
            for j in range(0, parameters.chain_length ):
                gf_a[r][i][j] = conjugate(gf_r[r][j][i])
                
    if( parameters.hubbard_interaction == 0 ):
            warnings.warn('Dear future Declan,  This formula is not valid for the interacting case.')

    transmission = [ [ 0 for i in range( parameters.chain_length ) for r in range( parameters.steps ) ] ]
    warnings.warn('Dear future Declan,  This assumes that the coupling matrices are diagonal. Your sincerely, past Declan ')  
    for r in range(0, parameters.steps):
        for i in range(0 , parameters.chain_length ):
            for j in range(0 , parameters.chain_length ):
                for k in range(0 , parameters.chain_length ):
                    transmission[i][r]  = coupling_left[r][i][k] * gf_r[r][k][j] * coupling_right[r][j][j] * gf_a[r][j][i] 
    """
    fig = plt.figure()
    plt.plot( parameters.energy , transmission[i]  , color='red'  ) 
    plt.title("Transmission")
    plt.xlabel("energy")
    plt.ylabel("Transmission probability")  
    plt.show()     
    """
    
    
    
    trace = [ 0 for r in range(parameters.steps ) ]
    
    for r in range(0 , parameters.steps  ):
        for i in range(0 , parameters.chain_length ):
            trace[r] +=  2 * (fermi_function(parameters.energy[r] - parameters.voltage_l[voltage_step] ) - fermi_function(parameters.energy[r] - parameters.voltage_r[voltage_step] ) ) * transmission[i][r] #factor of 2 is due to spin up and down

    current = trace_integrate(trace) 
    
    return current
    
def analytic_current( right_se_r: List[List[List[complex]]], left_se_r: List[List[List[complex]]], voltage_step: int ):
    coupling_right = coupling_matrices( right_se_r)
    coupling_left = coupling_matrices( left_se_r)
    current = 0
    analytic_gf = analytic_gf_1site()    
    trace = [ 0 for i in range(parameters.steps)]
    for r in range(0 , parameters.steps ):
        trace[r] = 2 * coupling_left[r][0][0] * analytic_gf[r] * coupling_right[r][0][0] * conjugate(analytic_gf[r]) * ( fermi_function(parameters.energy[r] + parameters.voltage_l[voltage_step]) - fermi_function(parameters.energy[r] + parameters.voltage_r[voltage_step]))
    
    delta_energy = ( parameters.e_upper_bound - parameters.e_lower_bound ) / parameters.steps
    
    for r in range( 0 , parameters.steps ):                
        current += delta_energy * trace[r] / (parameters.pi * 2)
    return current   


def compare_analytic_gf(green_function_up: List[List[List[complex]]]):
    analytic_gf_1site()
    for i in range(0, parameters.chain_length):

        plt.plot( parameters.energy , [ e[i][i].real for e in green_function_up]  , color='red' , label='Real Green up' ) 
        plt.plot( parameters.energy , [ e[i][i].imag for e in green_function_up], color='blue', label='Imaginary Green function')
        #plt.plot( parameters.energy , [ e.imag for e in analytic_gf ], color='green', label='imaginary analytic' ) 
        #plt.plot( parameters.energy , [e.real for e in analytic_gf] , color='yellow' , label='real analytic') 

        #plt.plot(energy, dos_spin_up[i] , color='blue', label='spin up DOS' ) 
        #plt.plot(energy, dos_spin_down[i], color='red', label='spin down DOS')
#   
   # plt.plot(energy,analytic, color='tomato')
    #plt.plot(energy,analytic2, color='green')
    #plt.xlim(-1, 3)
    plt.title("Converged Green function")
    plt.legend(loc='upper left')
    plt.xlabel("energy")
    plt.ylabel("Green Function")  
    plt.show()     

def embedding_self_energy_retarded():# this gets the embeddign self energy from the text file created by the leads self energy.
    self_energy_left=[create_matrix( parameters.chain_length ) for r in range(parameters.steps)]
    self_energy_right=[create_matrix( parameters.chain_length )for r in range(parameters.steps)]
        
    f= open(r"C:\Users\user\Desktop\Green function code\Green's Function\embedding_self_energy.txt", "r")
    lines = f.read().split(',')  
    for r in range(0, parameters.steps ):  
            self_energy_right[r][-1][-1] = float(lines[3+r*5])+1j*float(lines[4+r*5])
            self_energy_left[r][0][0] = float(lines[1+r*5])+1j*float(lines[2+r*5])  
    f.close() 

    return self_energy_left , self_energy_right

def current_voltage_graph():#this will create a current vs voltage graph for several potential biases. 
    points = 10
    current = [ 0 for i in range(points)]
    current_landauer = [ 0 for i in range(points)]

    voltage = [parameters.voltage_l[i] - parameters.voltage_r[i] for i in range(points) ]
    for i in range(0 , 1):#this for loop determines how many voltage biases we want ot consider.
        print("The bias is ", parameters.voltage_l[i] - parameters.voltage_r[i])
        self_energy = leads_self_energy.SelfEnergy(1 , i)    
        #this recalculates the embedding self energy for each bias.
        green_function_up, green_function_down, spectral_function_up, spectral_function_down, spin_up_occup, spin_down_occup , gf_int_lesser_up = gf_dmft(i)#we then obtain the gf and stuff for every bias. 
        #magnetisation=[spin_up_occup[i]-spin_down_occup[i] for i in range(0,parameters.chain_length)]

        #if ( parameters.hubbard_interaction == 0):
            #compare_analytic_gf(green_function_up)

        self_energy_left , self_energy_right = embedding_self_energy_retarded()
        
        if( parameters.hubbard_interaction == 0 ):
            current_landauer[i] = landauer_current(green_function_up, self_energy_left , self_energy_right , i ) 
            print("The landauer current is " , current_landauer[i], "The left voltage is " , parameters.voltage_l[i] , "The right voltage is " , parameters.voltage_r[i])
        
        current[i] = current_Meir_wingreen( spectral_function_up , gf_int_lesser_up, self_energy_left , self_energy_right , i) 
        print("The Meir_wingreen current is " , current[i], "The left voltage is " , parameters.voltage_l[i] , "The right voltage is " , parameters.voltage_r[i])
        print(" ")
        
    print(current, current_landauer)
    fig = plt.figure()
    plt.plot( voltage , current , color='blue' ) 
    plt.plot( voltage , current_landauer , color='red' )     
    plt.title("Current vs Voltage")
    #plt.legend(loc='upper right')
    plt.xlabel("Voltage")
    plt.ylabel("Current")  
    plt.show()
    
    return green_function_up


def main():
    time_start = time.perf_counter()
    
    green_function_up = current_voltage_graph()    
    #this outputs the green function to a text file if needed.
    f = open(r"C:\Users\user\Desktop\Green function code\Green's Function\green_function_not_FD.txt", "w")
    for r in range(0, parameters.steps ):

            f.write(str(green_function_up[r][0][0].real ))
            f.write( "," )          
            f.write(str(green_function_up[r][0][0].imag ))
            f.write( "," )

        #print("the number of columns should be " , r+1)
    f.close()
    time_elapsed = (time.perf_counter() - time_start)
    print(" The time it took the computation is" , time_elapsed)
    
if __name__=="__main__":#this will only run if it is a script and not a import module
    main()

