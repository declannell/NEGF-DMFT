import scipy.linalg as la
import matplotlib.pyplot as plt
import parameters

energypoints1 = 401
energy1 = [parameters.e_lower_bound+(parameters.e_upper_bound -
                                     parameters.e_lower_bound) / energypoints1 * x for x in range(energypoints1)]
energypoints2 = 161
energy2 = [parameters.e_lower_bound+(parameters.e_upper_bound -
                                     parameters.e_lower_bound) / energypoints2 * x for x in range(energypoints2)]

energypoints3 = 161
energy3 = [parameters.e_lower_bound+(parameters.e_upper_bound -
                                     parameters.e_lower_bound) / energypoints2 * x for x in range(energypoints3)]


se_200_k_401_e_points = [[0 for i in range(0, parameters.chain_length)]
                   for z in range(0, energypoints1)]
se_50_k_161_e_points = [[0 for i in range(0, parameters.chain_length)]
                   for z in range(0, energypoints2)]
se_80_k_161_e_points = [[0 for i in range(0, parameters.chain_length)]
                   for z in range(0, energypoints2)]

lines_complex = [[0, 0] for r in range(0, energypoints1)]
f = open('/home/declan/green_function_code/green_function/textfiles/local_se_200_k_points_401_energy.txt', 'r')
lines = f.read().rsplit()
for r in range(0, energypoints1):
    lines_complex[r] = lines[r].split(',')
    se_200_k_401_e_points[r][0] = float(
        lines_complex[r][0]) + 1j * float(lines_complex[r][1])
f.close()

lines_complex = [[0, 0] for r in range(0, energypoints2)]
f = open('/home/declan/green_function_code/green_function/textfiles/local_se_50_k_points_161_energy.txt', 'r')
lines = f.read().rsplit()
for r in range(0, energypoints2):
    lines_complex[r] = lines[r].split(',')
    se_50_k_161_e_points[r][0] = float(
        lines_complex[r][0]) + 1j * float(lines_complex[r][1])
f.close()

lines_complex = [[0, 0] for r in range(0, energypoints3)]
f = open('/home/declan/green_function_code/green_function/textfiles/local_se_80_k_points_161_energy.txt', 'r')
lines = f.read().rsplit()
for r in range(0, energypoints2):
    lines_complex[r] = lines[r].split(',')
    se_80_k_161_e_points[r][0] = float(
        lines_complex[r][0]) + .0072 + 1j * float(lines_complex[r][1])
f.close()

fig = plt.figure()
plt.plot(energy1, [
    e[0].real for e in se_200_k_401_e_points], color='blue', label='Real self energy 200 k points')

plt.plot(energy2, [
    e[0].real for e in se_50_k_161_e_points], color='red', label='Real self energy 50 k points')

plt.plot(energy3, [
    e[0].real for e in se_80_k_161_e_points], color='green', label='Real self energy 80 k points')
plt.title('The local self energy')
plt.legend(loc='upper left')
plt.xlabel("energy")
plt.ylabel("Self Energy")
plt.show()

fig = plt.figure()
plt.plot(energy1, [
    e[0].imag for e in se_200_k_401_e_points], color='blue', label='Imaginary self energy 200 k points')

plt.plot(energy2, [
    e[0].imag for e in se_50_k_161_e_points], color='red', label='Imaginary self energy 50 k points')

plt.plot(energy3, [
    e[0].imag for e in se_80_k_161_e_points], color='green', label='Imaginary self energy 80 k points')
plt.title('The local self energy')
plt.legend(loc='upper left')
plt.xlabel("energy")
plt.ylabel("Self Energy")
plt.show()
