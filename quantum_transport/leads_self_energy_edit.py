import matplotlib.pyplot as plt
import math
import numpy as np
import parameters
import warnings

class SelfEnergy: 
    self_energy_left: None
    self_energy_right: None
    self_energy_left_lesser: None
    self_energy_right_lesser: None
    surface_gf_r: None
    surface_gf_l: None
    transfer_matrix_l: None
    transfer_matrix_r: None
    chain_length_y: float
    principal_layer: float#this gives the number of orbitals within a principe layer
    parameters: None
    voltage_step : float
     
    def __init__(self , _principal_layer , _voltage_step):
        self.voltage_step = _voltage_step
        self.principal_layer = _principal_layer         
        self.h_s = create_matrix(self.principal_layer)
        self.surface_gf_l = [[create_matrix(self.principal_layer) for i in range(0, parameters.chain_length_y)] for r in range(0, parameters.steps )]
        self.surface_gf_r = [[create_matrix(self.principal_layer) for i in range(0, parameters.chain_length_y)] for r in range(0, parameters.steps )]
        self.transfer_matrix_l = [create_matrix(self.principal_layer) for r in range(0, parameters.steps )]
        self.transfer_matrix_r = [create_matrix(self.principal_layer) for r in range(0, parameters.steps )]
        self.self_energy_left = [[create_matrix(self.principal_layer) for i in range(0, parameters.chain_length_y )] for r in range(0, parameters.steps )]
        self.self_energy_right = [[create_matrix(self.principal_layer) for i in range(0, parameters.chain_length_y )] for r in range(0, parameters.steps )]
        
        self.self_energy_left_lesser = [[create_matrix(self.principal_layer) for i in range(0, parameters.chain_length_y )] for r in range(0, parameters.steps )]
        self.self_energy_right_lesser = [[create_matrix(self.principal_layer) for i in range(0, parameters.chain_length_y )] for r in range(0, parameters.steps )]       
        self.get_self_energy()


    def get_self_energy(self):
        #this is based on the paper https://iopscience.iop.org/article/10.1088/0305-4608/14/5/016/meta
        k_y = [2 * parameters.pi * m / parameters.chain_length_y for m in range(0, parameters.chain_length_y)]     
        print("the y direction length is ", parameters.chain_length_y)
        print(k_y) 
        for i in range(0, parameters.chain_length_y):
            print(k_y[i])
            self.get_transfer_matrix( k_y[i])
            self.sgf(i,k_y[i])
            self.lead_self_energy(i)
            self.lesser_self_energy(i)
            self.text_file_retarded(i)
            self.text_file_lesser(i)
            
    def text_file_retarded(self ,num): #num is the number of k-points
        f = open('/home/declan/green_function_code/green_function/textfiles/embedding_self_energy.txt', 'w')

        for num in range(0, parameters.chain_length_y):
            for r in range(0, parameters.steps ):
            
                f.write(str(parameters.energy[r].real) )
                f.write( "," )
                f.write(str(self.self_energy_left[r][num][0][0].real ))
                f.write( "," )          
                f.write(str(self.self_energy_left[r][num][0][0].imag ))
                f.write( "," )
                f.write(str(self.self_energy_right[r][num][0][0].real ))
                f.write( "," )
                f.write(str(self.self_energy_right[r][num][0][0].imag ))
                f.write( "," )
            
            #This commented out self energy is the wide band limit
            """
            f.write(str(r) )
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
        f.close()

    def text_file_lesser(self ,num): #num is the number of k-points
        f = open('/home/declan/green_function_code/green_function/textfiles/embedding_self_energy_lesser.txt', 'w')
        for num in range(0, parameters.chain_length_y):
            for r in range(0, parameters.steps ):
            
                f.write(str(self.self_energy_left_lesser[r][num][0][0].real ))
                f.write( "," )          
                f.write(str(self.self_energy_left_lesser[r][num][0][0].imag ))
                f.write( "," )
                f.write(str(self.self_energy_right_lesser[r][num][0][0].real ))
                f.write( "," )
                f.write(str(self.self_energy_right_lesser[r][num][0][0].imag ))
                f.write( "," )
            
            #This commented out self energy is the wide band limit        
            """
            f.write(str(0))
            f.write( "," )          
            f.write(str(-4*fermi_function(parameters.energy[r].real - parameters.voltage_l[self.voltage_step] )))
            f.write( "," )
            f.write(str(0))
            f.write( "," )
            f.write(str(-4*fermi_function(parameters.energy[r].real - parameters.voltage_r[self.voltage_step] )))
            f.write( "," )
            """
            
        f.close()
    
    def lesser_self_energy(self , num):
        for r in range(0 , parameters.steps ):
            for i in range( 0 , parameters.chain_length_y ):#fluctautation dissapation theorem is valid
               self.self_energy_left_lesser[r][i][0][0] = - fermi_function( parameters.energy[r].real - parameters.voltage_l[self.voltage_step] ) * ( self.self_energy_left[r][num][0][0] - parameters.conjugate( self.self_energy_left[r][num][0][0] ) )
               self.self_energy_right_lesser[r][num][0][0] = - fermi_function( parameters.energy[r].real - parameters.voltage_r[self.voltage_step] ) * ( self.self_energy_right[r][num][0][0] - parameters.conjugate( self.self_energy_right[r][num][0][0] ) )

        fig = plt.figure()
        """"
        plt.plot( [ e.real for e in parameters.energy] , [e[num][0][0].imag for e in self.self_energy_left], color='blue', label='imaginary self energy' ) 
        plt.plot( [ e.real for e in parameters.energy]  , [e[num][0][0].real for e in self.self_energy_left] , color='red' , label='real self energy') 
        plt.title(" Numerical left Self Energy")
        plt.legend(loc='upper right')
        plt.xlabel("energy")
        plt.ylabel("Self Energy")  
        plt.show()
        
        fig = plt.figure()
        
        plt.plot( [ e.real for e in parameters.energy]  , [e[num][0][0].imag for e in self.self_energy_right], color='blue', label='imaginary self energy' ) 
        plt.plot( [ e.real for e in parameters.energy]  , [e[num][0][0].real for e in self.self_energy_right] , color='red' , label='real self energy') 
        plt.title(" Numerical right Self Energy")
        plt.legend(loc='upper right')
        plt.xlabel("energy")
        plt.ylabel("Self Energy")  
        plt.show()

        plt.plot( [ e.real for e in parameters.energy]  , [e[num][0][0].imag for e in self.self_energy_left_lesser] , color = 'green'  , label='imaginary')
        plt.plot([ e.real for e in parameters.energy] , [e[num][0][0].real for e in self.self_energy_left_lesser] , color = 'orange'  , label='real')
        plt.title(" Numerical left Self Energy lesser")
        plt.legend(loc='upper right')
        plt.xlabel("energy")
        plt.ylabel("Self Energy lesser")  
        plt.show()
        
        plt.plot([ e.real for e in parameters.energy] , [e[num][0][0].imag for e in self.self_energy_right_lesser] , color = 'green'  , label='imaginary')
        plt.plot([ e.real for e in parameters.energy] , [e[num][0][0].real for e in self.self_energy_right_lesser] , color = 'orange'  , label='real')
        plt.title(" Numerical right Self Energy lesser")
        plt.legend(loc='upper right')
        plt.xlabel("energy")
        plt.ylabel("Self Energy lesser")  
        plt.show()
        """
        """
        for r in range(0 , parameters.steps ):
            for i in range( 0 , parameters.chain_length_y ):#fluctautation dissapation theorem is valid
               self.self_energy_left_lesser[r][i] = fermi_function( parameters.energy[r].real + parameters.voltage_l[self.voltage_step] ) * ( self.self_energy_left[r][num][0][0] - np.conjugate( self.self_energy_left[r][num][0][0] ) )
               self.self_energy_right_lesser[r][i] = fermi_function( parameters.energy[r].real + parameters.voltage_r[self.voltage_step] ) * ( self.self_energy_right[r][num][0][0] - np.conjugate( self.self_energy_right[r][num][0][0] ) )
        print(self.self_energy_left_lesser)
        """
        
        
    def get_transfer_matrix(self,k_y): #this assume t and t_tilde are the same and principal_layer is 1 atom thick
        t_next_l = [0 for r in range( 0 , parameters.steps )]
        t_next_r = [0 for r in range( 0 , parameters.steps )]
        a_l = [0 for r in range( 0 , parameters.steps )]
        a_r = [0 for r in range( 0 , parameters.steps )]        
        
        if( parameters.chain_length_y == 1 ):
            for r in range( 0 , parameters.steps ):                
                t_next_l[r] = parameters.hopping_lz / (parameters.energy[r] - parameters.onsite_l - parameters.voltage_l[self.voltage_step] )
                t_next_r[r] = parameters.hopping_rz / (parameters.energy[r] - parameters.onsite_r - parameters.voltage_r[self.voltage_step] )
                a_l[r] = t_next_l[r]
                a_r[r] = t_next_r[r]
                self.transfer_matrix_l[r][0][0] = t_next_l[r]
                self.transfer_matrix_r[r][0][0] = t_next_r[r]
                
        else:
            #print(-self.parameters.onsite_l-2*self.parameters.hopping_ly*np.cos(k_y))
            for r in range( 0 , parameters.steps ):
                t_next_l[r] = parameters.hopping_lz /( parameters.energy[r] - parameters.onsite_l - parameters.voltage_l[self.voltage_step] - 2 * parameters.hopping_ly * np.cos(k_y))
                t_next_r[r] = parameters.hopping_rz /( parameters.energy[r] - parameters.onsite_r - parameters.voltage_r[self.voltage_step] - 2 * parameters.hopping_ry * np.cos(k_y))
                a_l[r] = t_next_l[r]
                a_r[r] = t_next_r[r]

                self.transfer_matrix_l[r][0][0] = t_next_l[r]
                self.transfer_matrix_r[r][0][0] = t_next_r[r]   
                
        n = self.principal_layer**2 * parameters.steps
        differencelist = [ 0 for i in range(0 , 2 * n) ]
        old_transfer = [ 0 for z in range(0 , parameters.steps) ] 
        difference = 1       
        count = 0
        while( difference > 0.01 ):
            count += 1
            for r in range( 0 , parameters.steps ):
                t_next_l[r] = t_next_l[r]**2 / (1 -2 * t_next_l[r]**2)
                t_next_r[r] = t_next_r[r]**2 / (1 -2 * t_next_r[r]**2)
                a_l[r] = a_l[r] * t_next_l[r]
                a_r[r] = a_r[r] * t_next_r[r]
                self.transfer_matrix_l[r][0][0] = self.transfer_matrix_l[r][0][0] + a_l[r]
                self.transfer_matrix_r[r][0][0] = self.transfer_matrix_r[r][0][0] + a_r[r]
                
            for r in range( 0 , parameters.steps ):                
                differencelist[r] = abs(self.transfer_matrix_l[r][0][0].real - old_transfer[r].real)
                differencelist[n + r] = abs(self.transfer_matrix_l[r][0][0].imag - old_transfer[r].imag)
                old_transfer[r] = self.transfer_matrix_l[r][0][0]
            difference = max (differencelist)
            #print ("The difference is ", difference)
        #print(" This converged in " ,count, " iterations.\n")

    def sgf(self , num , k_y):
        if(parameters.chain_length_y == 1):
            for r in range(0 , parameters.steps):
                self.surface_gf_l[r][num][0][0] = 1 / (parameters.energy[r] - parameters.onsite_l - parameters.voltage_l[self.voltage_step]- parameters.hopping_lz * self.transfer_matrix_l[r][0][0] )
                self.surface_gf_r[r][num][0][0] = 1 / (parameters.energy[r] - parameters.onsite_r - parameters.voltage_r[self.voltage_step] - parameters.hopping_rz * self.transfer_matrix_r[r][0][0] )       
        else:
            print(parameters.onsite_r - 2 * parameters.hopping_ry * np.cos(k_y))
            for r in range(0 , parameters.steps):
                self.surface_gf_l[r][num][0][0] = 1 / (parameters.energy[r] - parameters.voltage_l[self.voltage_step] - parameters.onsite_l - 2 * parameters.hopping_ly * np.cos(k_y) - parameters.hopping_lz * self.transfer_matrix_l[r][0][0] )
                self.surface_gf_r[r][num][0][0] = 1 / (parameters.energy[r] - parameters.voltage_r[self.voltage_step] - parameters.onsite_r - 2 * parameters.hopping_ry * np.cos(k_y) - parameters.hopping_rz * self.transfer_matrix_r[r][0][0] )
    def lead_self_energy(self, num):
        for r in range(0,  parameters.steps):
            self.self_energy_left[r][num][0][0] = parameters.hopping_lc **2 * self.surface_gf_l[r][num][0][0]
            self.self_energy_right[r][num][0][0] = parameters.hopping_rc **2 * self.surface_gf_r[r][num][0][0]

def create_matrix(size):
    return [[0.0 for x in range(size)] for y in range(size)]

def fermi_function(energy):
    if( parameters.temperature == 0):
        if(energy < parameters.chemical_potential):
            return 1
        else:
            return 0
    else:
        return 1 / (1 + math.exp((energy - parameters.chemical_potential) / parameters.temperature))
                
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
    
def analytic_se(voltage_step):
    analytic_se = [0 for i  in range(parameters.steps)]# this assume the interaction between the scattering region and leads is nearest neighbour 
    for i in range(0 , parameters.steps):
        x = ( parameters.energy()[i].real - parameters.onsite_l - parameters.voltage_l[voltage_step] ) / ( 2 * parameters.hopping_lz )
        #print(x, energy[i])
        analytic_se[i] = (parameters.hopping_lc() ** 2) * (1 / parameters.hopping_lz) * x 
        if (abs(x) > 1):
            analytic_se[i] = analytic_se[i] - (parameters.hopping_lc() ** 2) * (1 / parameters.hopping_lz) * (sgn(x) * np.sqrt(abs(x) * abs(x) - 1)) 
        elif( abs(x) < 1):
            analytic_se[i] = analytic_se[i] - (parameters.hopping_lc() ** 2) * abs((1 / parameters.hopping_lz )) * (1j * np.sqrt(1 - abs(x) * abs(x)))

    #print(analytic_se)
  
    plt.plot( parameters.energy, [e.imag for e in analytic_se], color='blue', label='imaginary self energy' ) 
    plt.plot( parameters.energy, [e.real for e in analytic_se], color='red', label='real self energy') 
    plt.title(" Analytical left Self Energy")
    plt.legend(loc='upper right')
    plt.xlabel("energy")
    plt.ylabel("Self Energy") 
    plt.show()
    

def main():
    principal_layer = 1 
    voltage_step = parameters.voltage_step  # voltage step of zero is equilibrium.
    print(parameters.voltage_r[voltage_step])
    self_energy = SelfEnergy( principal_layer, voltage_step )


if __name__=="__main__":#this will only run if it is a script and not a import module
    main()