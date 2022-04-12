import scipy.linalg as la
import matplotlib.pyplot as plt
import math
import time
import leads_self_energy
import noninteracting_gf
import parameters
import warnings
from typing import List
import argparse


class Interacting_GF:
    kx: float
    ky: float
    voltage_step: float
    hamiltonian: List[List[complex]]
    interacting_gf: List[List[List[complex]]]

    def __init__(self, _kx: float, _ky: float, _voltage_step: int, self_energy_many_body):
        self.kx = _kx
        self.ky = _ky
        #print("The value of kx is ", self.kx, "The value of ky is ", self.ky)
        self.voltage_step = _voltage_step
        self.hamiltonian = create_matrix(parameters.chain_length)
        self.interacting_gf = [create_matrix(
            parameters.chain_length) for r in range(parameters.steps)]
        # this willgetting the embedding self energies from the leads code
        self.get_hamiltonian()
        self.get_interacting_gf(self_energy_many_body)

    def get_hamiltonian(self):
        # self_energy.plot_self_energy()
        for i in range(0, parameters.chain_length-1):
            self.hamiltonian[i][i+1] = parameters.hopping
            self.hamiltonian[i+1][i] = parameters.hopping
        for i in range(0, parameters.chain_length):
            voltage_i = parameters.voltage_l[self.voltage_step] - (i + 1) / (float)(parameters.chain_length + 1) * (
                parameters.voltage_l[self.voltage_step] - parameters.voltage_r[self.voltage_step])
            #print("The external voltage is on site ",  i, " is ", voltage_i)
            self.hamiltonian[i][i] = parameters.onsite + 2 * parameters.hopping_x * \
                math.cos(self.kx) + 2 * parameters.hopping_y * \
                math.cos(self.ky) + voltage_i

        """    
        plt.plot(parameters.energy, [e[0][0].real for e in self.effective_hamiltonian], color='red', label='real effective hamiltonian') 
        plt.plot(parameters.energy, [e[0][0].imag for e in self.effective_hamiltonian], color='blue', label='Imaginary effective hamiltonian')
        plt.title("effective hamiltonian")
        plt.legend(loc='upper left')
        plt.xlabel("energy")
        plt.ylabel("effective hamiltonian")  
        plt.show()   
        """

    def get_interacting_gf(self, self_energy_many_body):
        inverse_green_function = create_matrix(parameters.chain_length)
        self_energy = leads_self_energy.EmbeddingSelfEnergy(
            self.kx, self.ky, parameters.voltage_step)

        for r in range(0, parameters.steps):
            if parameters.chain_length != 1:
                inverse_green_function[0][0] = - self_energy.self_energy_left[r]
                inverse_green_function[-1][-1] = - self_energy.self_energy_right[r]
            elif parameters.chain_length == 1:
                inverse_green_function[0][0] = - 2 * self_energy.self_energy_left[r]

            for i in range(0, parameters.chain_length):
                for j in range(0, parameters.chain_length):
                    if (i == j):
                        inverse_green_function[i][j] += parameters.energy[r] - \
                            self.hamiltonian[i][j] - \
                            self_energy_many_body[r][i]
                    else:
                        inverse_green_function[i][j] = - \
                            self.hamiltonian[r][i][j]

            self.interacting_gf[r] = la.inv(
                inverse_green_function, overwrite_a=False, check_finite=True)

    def plot_greenfunction(self):
        for i in range(0, parameters.chain_length):

            plt.plot(parameters.energy, [
                     e[i][i].real for e in self.interacting_gf], color='red', label='Real Green up')
            plt.plot(parameters.energy, [
                     e[i][i].imag for e in self.interacting_gf], color='blue', label='Imaginary Green function')
            j = i + 1
            plt.title('Noninteracting Green function site %i' % j)
            plt.legend(loc='upper left')
            plt.xlabel("energy")
            plt.ylabel("Noninteracting green Function")
            plt.show()

    # this allows me to print the effective hamiltonian if called for a certain energy point specified by num.
    def print_hamiltonian(self):
        # eg. hamiltonian.print(4) will print the effective hamiltonian of the 4th energy step
        for i in range(0, parameters.chain_length):
            # rjust adds padding, join connects them all
            row_string = " ".join((str(r).rjust(5, " ")
                                  for r in self.hamiltonian[i]))
            print(row_string)


# this should work as in first order interaction, it gives the same result as fluctuation dissaption thm to 11 deciaml places
def get_spin_occupation(gf_lesser_up: List[complex], gf_lesser_down: List[complex]):
    delta_energy = (parameters.e_upper_bound -
                    parameters.e_lower_bound)/parameters.steps
    result_up, result_down = 0, 0
    for r in range(0, parameters.steps):
        result_up = (delta_energy) * gf_lesser_up[r] + result_up
        result_down = (delta_energy) * gf_lesser_down[r] + result_down
    x = -1j / (2 * parameters.pi) * result_up
    y = -1j / (2 * parameters.pi) * result_down
    return x, y


def integrate(gf_1: List[complex], gf_2: List[complex], gf_3: List[complex], r: int):
    # in this function, the green functions are 1d arrays in energy. this is becasue we have passed the diagonal component of the green fun( lesser, or retarded).The
    delta_energy = (parameters.e_upper_bound -
                    parameters.e_lower_bound) / parameters.steps
    result = 0
    for i in range(0, parameters.steps):
        for j in range(0, parameters.steps):
            if (((i + j - r) >= 0) and ((i + j - r) < parameters.steps)):
                # this integrates like PHYSICAL REVIEW B 74, 155125 2006
                # I say the green function is zero outside e_lower_bound and e_upper_bound. This means I need the final green function in the integral to be within an energy of e_lower_bound
                # and e_upper_bound. The index of 0 corresponds to e_lower_bound. Hence we need i+J-r>0 but in order to be less an energy of e_upper_bound we need i+j-r<steps. These conditions enesure the enrgy of the gf3 greens function to be within (e_upper_bound, e_lower_bound)
                result = (delta_energy / (2 * parameters.pi)) ** 2 * \
                    gf_1[i] * gf_2[j] * gf_3[i+j-r] + result
            else:
                result = result
    return result


# this creates the entire parameters.energy() array at once
def self_energy_2nd_order(impurity_gf_up: List[complex], impurity_gf_down: List[complex], impurity_gf_up_lesser: List[complex], impurity_gf_down_lesser: List[complex]):
    impurity_self_energy = [0 for z in range(0, parameters.steps)]

    # the are calculating the self parameters.energy() sigma_{ii}(E) for each discretized parameters.energy(). To do this we pass the green_fun_{ii} for all energies as we need to integrate over all energies in the integrate function
    for r in range(0, parameters.steps):
        impurity_self_energy[r] = parameters.hubbard_interaction ** 2 * (integrate(
            impurity_gf_up, impurity_gf_down, impurity_gf_down_lesser, r))  # line 3
        impurity_self_energy[r] += parameters.hubbard_interaction ** 2 * (integrate(
            impurity_gf_up, impurity_gf_down_lesser, impurity_gf_down_lesser, r))  # line 2
        impurity_self_energy[r] += parameters.hubbard_interaction ** 2 * (integrate(
            impurity_gf_up_lesser, impurity_gf_down, impurity_gf_down_lesser, r))  # line 1
        impurity_self_energy[r] += parameters.hubbard_interaction ** 2 * (integrate(
            impurity_gf_up_lesser, impurity_gf_down_lesser, [parameters.conjugate(e) for e in impurity_gf_down], r))  # line 4
    return impurity_self_energy

# this is only used to compare the lesser green functions using two different methods. This is not used in the calculation of the self energies.


def fluctuation_dissipation(green_function: List[complex]):
    g_lesser = [0 for z in range(0, parameters.steps)]
    for r in range(0, parameters.steps):
        g_lesser[r] = - parameters.fermi_function(parameters.energy[r].real) * (
            green_function[r] - parameters.conjugate(green_function[r]))
    return g_lesser


# coupling matirces for the current calculation.
def coupling_matrices(se_r: List[List[List[complex]]]):
    coupling_mat = [create_matrix(parameters.chain_length)
                    for r in range(parameters.steps)]
    for r in range(0, parameters.steps):
        for i in range(parameters.chain_length):
            for j in range(parameters.chain_length):
                coupling_mat[r][i][j] = 1j * \
                    (se_r[r][i][j] - parameters.conjugate(se_r[r][j][i]))
    return coupling_mat


def transmission(gf_r: Interacting_GF):  # this is the green function for each k point
    coupling_right = coupling_matrices(right_se_r)
    coupling_left = coupling_matrices(left_se_r)

    transmission = [[0 for i in range(parameters.chain_length)]
                    for r in range(parameters.steps)]
    warnings.warn(
        'Dear future Declan,  This assumes that the coupling matrices are diagonal. Your sincerely, past Declan ')
    for r in range(0, parameters.steps):
        for i in range(0, parameters.chain_length):
            for j in range(0, parameters.chain_length):
                for k in range(0, parameters.chain_length):
                    transmission[r][i] = coupling_left[r][i][k] * gf_r[r][k][j] * \
                        coupling_right[r][j][j] * \
                        parameters.conjugate(gf_r[r][i][j])

    return transmission

    """
    trace = [ 0 for r in range(parameters.steps ) ]
    
    for r in range(0 , parameters.steps  ):
        for i in range(0 , parameters.chain_length ):
            trace[r] +=  2 * (fermi_function(parameters.energy[r] - parameters.voltage_l[voltage_step] ) - fermi_function(parameters.energy[r] - parameters.voltage_r[voltage_step] ) ) * transmission[r][i] #factor of 2 is due to spin up and down

    current = trace_integrate(trace) 
    
    return current"""


def impurity_solver(impurity_gf_up: List[complex], impurity_gf_down: List[complex]):
    impurity_self_energy_up = [0 for z in range(0, parameters.steps)]
    impurity_self_energy_down = [0 for z in range(0, parameters.steps)]

    if (parameters.voltage_step == 0):
        impurity_gf_up_lesser = fluctuation_dissipation(impurity_gf_up)
        impurity_gf_down_lesser = fluctuation_dissipation(impurity_gf_down)

    impurity_spin_up, impurity_spin_down = get_spin_occupation(
        impurity_gf_up_lesser, impurity_gf_down_lesser)

    if (parameters.interaction_order == 2):
        impurity_self_energy_up = self_energy_2nd_order(
            impurity_gf_up, impurity_gf_down, impurity_gf_up_lesser, impurity_gf_down_lesser)
        impurity_self_energy_down = self_energy_2nd_order(
            impurity_gf_down, impurity_gf_up, impurity_gf_down_lesser, impurity_gf_up_lesser)
        for r in range(0, parameters.steps):
            impurity_self_energy_up[r] += parameters.hubbard_interaction * \
                impurity_spin_down
            impurity_self_energy_down[r] += parameters.hubbard_interaction * \
                impurity_spin_up

    if (parameters.interaction_order == 1):
        for r in range(0, parameters.steps):
            impurity_self_energy_up[r] = parameters.hubbard_interaction * \
                impurity_spin_down
            impurity_self_energy_down[r] = parameters.hubbard_interaction * \
                impurity_spin_up

    return impurity_self_energy_up, impurity_self_energy_down, impurity_spin_up, impurity_spin_down


def sum_gf_interacting(r, i, j, gf_interacting_up, gf_interacting_down):
    up = 0.0
    down = 0.0
    num_k_points = parameters.chain_length_x * parameters.chain_length_y
    for kx_i in range(0, parameters.chain_length_x):
        for ky_i in range(0, parameters.chain_length_y):
            up += (
                gf_interacting_up[ky_i][kx_i].interacting_gf[r][i][j]
                / num_k_points)
            down += (
                gf_interacting_down[ky_i][kx_i].interacting_gf[r][i][j]
                / num_k_points)
    return (up, down)


def create_matrix(size: int):
    return [[0.0 for x in range(size)] for y in range(size)]


def dmft(voltage: int, kx: List[float], ky: List[float]):
    self_energy_mb_up = [
        [0 for i in range(0, parameters.chain_length)]for z in range(0, parameters.steps)]
    self_energy_mb_down = [
        [0 for i in range(0, parameters.chain_length)]for z in range(0, parameters.steps)]

    n = parameters.chain_length**2 * parameters.steps
    differencelist = [0 for i in range(0, 2 * n)]
    old_green_function = [[[1.0 + 1j for x in range(parameters.chain_length)] for y in range(
        parameters.chain_length)] for z in range(0, parameters.steps)]
    difference = 100.0
    count = 0
    # these allows us to determine self consistency in the retarded green function
    while (difference > 0.0001 and count < 25):
        count += 1
        gf_interacting_up = [[Interacting_GF(kx[i], ky[j], voltage, self_energy_mb_up) for i in range(
            0, parameters.chain_length_x)] for j in range(0, parameters.chain_length_y)]
        gf_interacting_down = [[Interacting_GF(kx[i], ky[j], voltage, self_energy_mb_down) for i in range(
            0, parameters.chain_length_x)] for j in range(0, parameters.chain_length_y)]

        # this quantity is the green function which is averaged over all k points.
        gf_local_up = [create_matrix(parameters.chain_length)
                       for z in range(0, parameters.steps)]
        gf_local_down = [create_matrix(parameters.chain_length)
                         for z in range(0, parameters.steps)]

    # for r, i, j in cartesian(parameters.steps, parameters.chain_length, parameters.chain_length):

        for r in range(0, parameters.steps):
            for i in range(0, parameters.chain_length):
                for j in range(0, parameters.chain_length):
                    (up, down) = sum_gf_interacting(
                        r, i, j, gf_interacting_up, gf_interacting_down)
                    gf_local_up[r][i][j] += up
                    gf_local_down[r][i][j] += down

        # this will compare the new green function with the last green function for convergence
        for r in range(0, parameters.steps):
            for i in range(0, parameters.chain_length):
                for j in range(0, parameters.chain_length):
                    differencelist[r + i + j] = abs(
                        gf_local_up[r][i][j].real - old_green_function[r][i][j].real)
                    differencelist[n + r + i + j] = abs(
                        gf_local_up[r][i][j].imag - old_green_function[r][i][j].imag)
                    old_green_function[r][i][j] = gf_local_up[r][i][j]

        difference = max(differencelist)

        if (difference < 0.0001):
            break

        if(parameters.interaction_order != 0):
            for i in range(0, parameters.chain_length):
                impurity_self_energy_up, impurity_self_energy_down, spin_up_occup, spin_down_occup = (
                    impurity_solver([e[i][i] for e in gf_local_up], [e[i][i] for e in gf_local_down]))
                """
                print("The spin up occupancy for the site",
                      i + 1, " is ", spin_up_occup)
                print("The spin down occupancy for the site",
                      i + 1, " is ", spin_down_occup)
                """
                for r in range(0, parameters.steps):
                    self_energy_mb_up[r][i] = impurity_self_energy_up[r]
                    self_energy_mb_down[r][i] = impurity_self_energy_down[r]
        else:
            break
        print("The count is ", count, "The difference is ", difference)

    #local_transmission = transmission(gf_interacting_up)

    if (parameters.hubbard_interaction == 0.0 and parameters.voltage_step == 0):
        spin_up_occup, spin_down_occup = [0 for i in range(parameters.chain_length)], [
            0 for i in range(parameters.chain_length)]
        for i in range(0, parameters.chain_length):
            spin_up_occup[i], spin_down_occup[i] = first_order_self_energy(
                [e[i][i] for e in gf_local_up], [e[i][i] for e in gf_local_down])

    parser = argparse.ArgumentParser()

    parser.add_argument("-tf", "--textfile", help="Textfiles", type=bool)

    args = parser.parse_args()
    if(args.textfile == True):
        f = open('/home/declan/green_function_code/green_function/textfiles/local_gf_%i_k_points_%i_energy.txt' %
                 (parameters.chain_length_x, parameters.steps), 'w')
        for r in range(0, parameters.steps):
            f.write(str(gf_local_up[r][0][0].real))
            f.write(",")
            f.write(str(gf_local_up[r][0][0].imag))
            f.write(",")
            f.write("\n")
        f.close()

        if(parameters.hubbard_interaction != 0):
            f = open('/home/declan/green_function_code/green_function/textfiles/local_se_%i_k_points_%i_energy.txt' %
                     (parameters.chain_length_x, parameters.steps), 'w')
            for r in range(0, parameters.steps):
                f.write(str(self_energy_mb_up[r][0].real))
                f.write(",")
                f.write(str(self_energy_mb_up[r][0].imag))
                f.write("\n")
            f.close()

    for i in range(0, parameters.chain_length):
        plt.plot(parameters.energy, [
            e[i][i].real for e in gf_local_up], color='red', label='Real Green up')
        plt.plot(parameters.energy, [
            e[i][i].imag for e in gf_local_up], color='blue', label='Imaginary Green function')
        j = i + 1
        plt.title('The local Green function site % i for %i k points and %i energy points' % (
            j, parameters.chain_length_x, parameters.steps))
        plt.legend(loc='upper left')
        plt.xlabel("energy")
        if(parameters.hubbard_interaction == 0):
            plt.ylabel("Noninteracting green Function")
        else:
            plt.ylabel("Interacting green Function")
        plt.show()

    if(parameters.hubbard_interaction != 0):
        for i in range(0, parameters.chain_length):
            fig = plt.figure()
            plt.plot(parameters.energy, [
                e[i].imag for e in self_energy_mb_down], color='blue', label='imaginary self energy')
            #plt.plot(parameters.energy, [
                #e[i].real for e in self_energy_mb_down], color='red', label='real self energy')
            j = i + 1
            plt.title('The local self energy site % i (%i k %i energy points)' % (
                j, parameters.chain_length_x, parameters.steps))
            plt.legend(loc='upper right')
            plt.xlabel("energy")
            plt.ylabel("Self Energy")
            plt.show()

        for i in range(0, parameters.chain_length):
            fig = plt.figure()
            plt.plot(parameters.energy, [
                e[i].imag for e in self_energy_mb_up], color='blue', label='imaginary self energy')
            plt.plot(parameters.energy, [
                e[i].real for e in self_energy_mb_up], color='red', label='real self energy')
            j = i + 1
            plt.title('Many-body self energy spin up site %i' % j)
            plt.legend(loc='upper right')
            plt.xlabel("energy")
            plt.ylabel("Self Energy")
            plt.show()

    #print("The spin up occupaton probability is ", spin_up_occup)
    #print("The spin down occupaton probability is ", spin_down_occup)
    # if(voltage == 0):#this compares the two methods in equilibrium
        #compare_g_lesser(gf_int_lesser_up , gf_int_up)
    for i in range(0, parameters.chain_length):
        print("The spin up occupancy for the site",
              i + 1, " is ", spin_up_occup)
        print("The spin down occupancy for the site",
              i + 1, " is ", spin_down_occup)
    print("The count is ", count)
    return gf_local_up, gf_local_down  # , spin_up_occup, spin_down_occup


def first_order_self_energy(gf_local_up: List[complex], gf_local_down: List[complex]):
    gf_up_lesser = fluctuation_dissipation(gf_local_up)
    gf_down_lesser = fluctuation_dissipation(gf_local_down)
    spin_up_occup, spin_down_occup = get_spin_occupation(
        gf_up_lesser, gf_down_lesser)
    return spin_up_occup, spin_down_occup


# this the analytic soltuion for the noninteracting green function when we have a single site in the scattering region
def analytic_local_gf_1site(gf_int_up: List[List[List[complex]]], kx: List[float], ky: List[float]):
    # this assume the interaction between the scattering region and leads is nearest neighbour
    analytic_gf = [0 for i in range(parameters.steps)]
    for i in range(0, parameters.chain_length_x):
        for j in range(0, parameters.chain_length_y):
            self_energy = leads_self_energy.EmbeddingSelfEnergy(
                kx[i], ky[j], parameters.voltage_step)
            #self_energy.plot_self_energy()
            num_k_points = parameters.chain_length_x * parameters.chain_length_y
            for r in range(0, parameters.steps):
                x = parameters.energy[r].real - parameters.onsite - 2 * parameters.hopping_x * math.cos(kx[i]) \
                    - 2 * parameters.hopping_y * math.cos(ky[j]) - self_energy.self_energy_left[r].real - \
                    self_energy.self_energy_right[r].real
                y = self_energy.self_energy_left[r].imag + \
                    self_energy.self_energy_right[r].imag
                analytic_gf[r] += 1 / num_k_points * (x / (x * x + y * y) + \
                    1j * y / (x * x + y * y))

    #plt.plot(parameters.energy, [
            # e[0][0].real for e in gf_int_up], color='red', label='real green function')
    plt.plot(parameters.energy, [
             -e[0][0].imag for e in gf_int_up], color='blue', label='imaginary green function')
    plt.plot(parameters.energy, [-e.imag for e in analytic_gf],
             color='blue', label='analytic imaginary green function')
    #plt.plot(parameters.energy, [e.real for e in analytic_gf],
             #color='red', label='analytic real green function')
    plt.title(" Analytical Green function and numerical GF")
    #plt.legend(loc='upper right')
    plt.xlabel("energy")
    plt.ylabel("Green Function")
    plt.show()


def main():
    kx = [0 for m in range(1, parameters.chain_length_x + 1)]
    ky = [0 for m in range(1, parameters.chain_length_y + 1)]
    for i in range(0, parameters.chain_length_y):
        if (parameters.chain_length_y != 1):
            ky[i] = 2 * parameters.pi * i / parameters.chain_length_y + 1
        elif (parameters.chain_length_y == 1):
            ky[i] = parameters.pi / 2.0

    for i in range(0, parameters.chain_length_x):
        if (parameters.chain_length_x != 1):
            kx[i] = 2 * parameters.pi * i / parameters.chain_length_x + 1
        elif (parameters.chain_length_x == 1):
            kx[i] = parameters.pi / 2.0

    # voltage step of zero is equilibrium.
    print("The voltage difference is ",
          parameters.voltage_l[parameters.voltage_step] - parameters.voltage_r[parameters.voltage_step])
    print("The number of sites in the z direction is ", parameters.chain_length)
    print("The number of sites in the x direction is ", parameters.chain_length_x)
    print("The number of sites in the y direction is ", parameters.chain_length_y)
    #print("The ky value is ", ky)
    #print("The kx value is ", kx)
    time_start = time.perf_counter()
    green_function_up, green_function_down = dmft(
        parameters.voltage_step, kx, ky)
    if (parameters.chain_length ==1 and parameters.hubbard_interaction == 0):
        analytic_local_gf_1site(green_function_up, kx, ky)

    for i in range(0, parameters.chain_length):
        plt.plot(parameters.energy, [
            -e[i][i].imag for e in green_function_up], color='blue', label='Imaginary Green function')
        j = i + 1
        plt.title('The local Green function site % i for %i k points and %i energy points' % (
            j, parameters.chain_length_x, parameters.steps))
        plt.legend(loc='upper left')
        plt.xlabel("energy")
        if(parameters.hubbard_interaction == 0):
            plt.ylabel("Noninteracting green Function")
        else:
            plt.ylabel("Interacting green Function")
        plt.show()
    time_elapsed = (time.perf_counter() - time_start)
    print(" The time it took the computation is", time_elapsed)


if __name__ == "__main__":  # this will only run if it is a script and not a import module
    main()
