import scipy.linalg as la
import matplotlib.pyplot as plt
import numpy as np
import math
from dataclasses import dataclass
import time

@dataclass #creates init function for me
class Parameters:
    onsite: float
    gamma: float
    hopping_lx: float
    hopping_ly: float
    hopping_ld: float
    chain_length_x: int
    chain_length_y: int
    chemical_potential: float
    temperature: float
    steps: float
    e_upper_bound: float
    e_lower_bound: float
    hubbard_interaction: float

class SelfEnergy: 
    self_energy_left: None
    self_energy_right: None
    surface_gf_r: None
    surface_gf_l: None
    transfer_matrix: None
    h_01_l: None
    h_01_r: None
    energy: None
    h_s: None
    chain_length_y: float
    principal_layer: float#this gives the number of orbitals within a principe layer
    parameters: None
    
    
    
    def __init__(self, parameters, _energy, _principal_layer):
        
        self.principal_layer=_principal_layer 
        self.parameters=parameters
        self.energy=_energy
        
        self.h_s=create_matrix(self.principal_layer)
        self.h_01_l=create_matrix(self.principal_layer)
        self.h_01_r=create_matrix(self.principal_layer)
        self.surface_gf_l=[[create_matrix(self.principal_layer) for i in range(0, parameters.chain_length_y)] for r in range(0, parameters.steps)]
        self.transfer_matrix=[create_matrix(self.principal_layer) for r in range(0, parameters.steps)]
        self.self_energy_left=[[create_matrix(self.principal_layer) for i in range(0, parameters.chain_length_y)] for r in range(0, parameters.steps)]
        self.self_energy_right=[[create_matrix(self.principal_layer) for i in range(0, parameters.chain_length_y)] for r in range(0, parameters.steps)]
        
        self.get_self_energy()
        


            #difference=max(differencelist)
        #print("The difference is " , difference)
    def get_self_energy(self):
        k_y=[ 2*np.pi*m/self.parameters.chain_length_y for m in range(0,self.parameters.chain_length_y)]      
        for i in range(0, self.parameters.chain_length_y):
            #self.assign_hamiltonian(k_y[i])    
            #self.print(self.h_s)
            #self.print(self.h_01_l)
            self.get_transfer_matrix( k_y[i])#this could be changed so i dont need to store it
            self.sgf(i,k_y[i])
            self.lead_self_energy(i)
            self.plot(i)
            self.text_file(i)
            
        
    def assign_hamiltonian(self,k_y):
        if(self.parameters.chain_length_y==1):
            for i in range(0,self.principal_layer):
                self.h_s[i][i]=self.parameters.onsite
        
        else:
            for i in range(0,self.principal_layer):
                self.h_s[i][i]=self.parameters.onsite+2*self.parameters.hopping_ly*np.cos(k_y)
            
        for i in range(0, self.principal_layer):
            for j in range(0,self.principal_layer):
                self.h_01_l[i][j]=self.parameters.hopping_lx
                self.h_01_r[i][j]=self.parameters.hopping_lx
        for i in range(0,self.principal_layer-1):
            self.h_s[i+1][i]=self.parameters.hopping_lx
            self.h_s[i][i+1]=self.parameters.hopping_lx 
        
            
    def print(self, matrix):
        for i in range(self.principal_layer):
            row_string = " ".join((str(r).rjust(5, " ") for r in matrix[i])) #rjust adds padding, join connects them all
            print(row_string)
    
    def text_file(self ,num): #num is the number of k-points
        f = open(r"C:\Users\user\Desktop\Green function code\Green's Function\embedding_self_energy.txt", "w")
        for r in range(0, self.parameters.steps):
            #print(self.parameters.steps)
            
            f.write(str(self.energy[r].real) )
            f.write( "," )
            f.write(str(self.self_energy_left[r][num][0][0].real ))
            f.write( "," )          
            f.write(str(self.self_energy_left[r][num][0][0].imag ))
            f.write( "," )
            f.write(str(self.self_energy_right[r][num][0][0].real ))
            f.write( "," )
            f.write(str(self.self_energy_right[r][num][0][0].imag ))
            f.write( "," )
            """
            f.write(str(self.energy[r].real) )
            f.write( "," )
            f.write(str(0))
            f.write( "," )          
            f.write(str(-2))
            f.write( "," )
            f.write(str(0))
            f.write( "," )
            f.write(str(-2))
            f.write( "," )
            """
            
        print("the number of columns should be " , r+1)
        f.close()

    
    def plot(self, num):
        fig = plt.figure()
        
        plt.plot(self.energy, [e[num][0][0].imag for e in self.self_energy_left], color='blue', label='imaginary self energy' ) 
        plt.plot(self.energy, [e[num][0][0].real for e in self.self_energy_left] , color='red' , label='real self energy') 
        plt.title(" Numerical Self Energy")
        plt.legend(loc='upper right')
        plt.xlabel("energy")
        plt.ylabel("Self Energy")  
        plt.show()

    def get_transfer_matrix(self,k_y): #this assume t and t_tilde are the same and principal_layer is 1 atom thick
        t_0=[0 for r in range(0,self.parameters.steps)]
        t_next=[0 for r in range(0,self.parameters.steps)]
        a=[0 for r in range(0,self.parameters.steps)]
        
        if(self.parameters.chain_length_y==1):
            for r in range(0,self.parameters.steps):                
                t_next[r]=self.parameters.hopping_lx/(self.energy[r]-self.parameters.onsite)
                a[r]=t_next[r]
                self.transfer_matrix[r][0][0]=t_next[r]
                
        else:
            #print(-self.parameters.onsite-2*self.parameters.hopping_ly*np.cos(k_y))
            for r in range(0,self.parameters.steps):
                t_0[r]=self.parameters.hopping_lx/(self.energy[r]-self.parameters.onsite-2*self.parameters.hopping_ly*np.cos(k_y))
                a[r]=t_0[r]
                t_next[r]=t_0[r]
                self.transfer_matrix[r][0][0]=t_0[r]
            
        n=self.principal_layer**2*self.parameters.steps
        differencelist=[0 for i in range(0,2*n)]
        old_transfer=[0 for z in range(0,self.parameters.steps)] 
        difference=1       
        count=0
        while(difference>0.01):
            count+=1
            for r in range(0,self.parameters.steps):
                t_next[r]=t_next[r]**2/(1-2*t_next[r]**2)
                a[r]=a[r]*t_next[r]
                self.transfer_matrix[r][0][0]=self.transfer_matrix[r][0][0]+a[r]

            for r in range(0,self.parameters.steps):                
                differencelist[r]=abs(self.transfer_matrix[r][0][0].real-old_transfer[r].real)
                differencelist[n+r]=abs(self.transfer_matrix[r][0][0].imag-old_transfer[r].imag)
                old_transfer[r]=self.transfer_matrix[r][0][0]
            difference=max(differencelist)
            print("The difference is ", difference)
        print(" This converged in " ,count, " iterations.\n")

    def sgf(self, num, k_y):
        if(self.parameters.chain_length_y==1):
            for r in range(0, self.parameters.steps):
                self.surface_gf_l[r][num][0][0]=1/(self.energy[r]-self.parameters.onsite-self.parameters.hopping_lx*self.transfer_matrix[r][0][0])
       
        else:
            for r in range(0, self.parameters.steps):
                self.surface_gf_l[r][num][0][0]=1/(self.energy[r]-self.parameters.onsite-2*self.parameters.hopping_ly*np.cos(k_y)-self.parameters.hopping_lx*self.transfer_matrix[r][0][0])

    def lead_self_energy(self, num):
        for r in range(0, self.parameters.steps):
            self.self_energy_left[r][num][0][0]=self.parameters.hopping_ld**2*self.surface_gf_l[r][num][0][0]
            self.self_energy_right[r][num][0][0]=self.parameters.hopping_ld**2*self.surface_gf_l[r][num][0][0]

def create_matrix(size):
    return [[0.0 for x in range(size)] for y in range(size)]


def matrix_mult(a,b, size):#assume both are square matrices
    c=create_matrix(size)
    for i in range(0, size):
        for j in range(0,size):
            for k in range(0,size):
                c[i][j]=a[i][k]*b[k][j]
                
def theta_function(a,b):
    if( a > b):
        return 1
    else:
        return 0
    
def sgn(x):
    if x>0 :
        return 1
    elif x<0 :
        return -1
    
def analytic_se(parameters, energy):
    analytic_se= [ 0 for i  in range(parameters.steps)]# this assume the interaction between the scattering region and leads is nearest neighbour 
    for i in range(0,parameters.steps):
        x=(energy[i].real-parameters.onsite)/(2*parameters.hopping_lx)
        #print(x, energy[i])
        analytic_se[i]=(parameters.hopping_ld**2) * (1/parameters.hopping_lx) * ( x  ) 
        if (abs(x) > 1):
            analytic_se[i] = analytic_se[i] - (parameters.hopping_ld**2) * (1/parameters.hopping_lx) * (  sgn(x) * np.sqrt(abs(x)*abs(x)-1) ) 
        elif( abs(x) < 1):
            analytic_se[i] = analytic_se[i] - (parameters.hopping_ld**2) * abs((1/parameters.hopping_lx)) * (1j*np.sqrt(1-abs(x)*abs(x) ))

    #print(analytic_se)
    plt.plot(energy, [ e.imag for e in analytic_se ], color='blue', label='imaginary self energy' ) 
    plt.plot(energy, [e.real for e in analytic_se] , color='red' , label='real self energy') 
    plt.title(" Analytical Self Energy")
    plt.legend(loc='upper right')
    plt.xlabel("energy")
    plt.ylabel("Self Energy")  
    plt.show()

def main():

    onsite, gamma, hopping_lx, hopping_ly , hopping_ld, chemical_potential, temperature , hubbard_interaction = 0.0 , 2.0 , -.5 , -0.5,-0.5,0.0, 0.0 , 0.3
    #The left and right self energies are not functions of the number of sites in the scattering region
    chain_length_y=1
    principal_layer=1
    steps=81 #number of energy points
    e_upper_bound , e_lower_bound = 10 , -10
    energy=[e_lower_bound+(e_upper_bound-e_lower_bound)/steps*x+0.0000001*1j  for x in range(steps)]
    parameters=Parameters(onsite, gamma, hopping_lx, hopping_ly, hopping_ld, 0 ,chain_length_y, chemical_potential, temperature, steps, e_upper_bound ,e_lower_bound, hubbard_interaction)
    self_energy = SelfEnergy(parameters, energy, principal_layer)
    #self_energy.print()
    analytic_se(parameters, energy)
    #print(energy)

if __name__=="__main__":#this will only run if it is a script and not a import module
    main()