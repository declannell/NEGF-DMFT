import scipy.linalg as la
import matplotlib.pyplot as plt
import parameters as par
import math
import argparse
parser = argparse.ArgumentParser()

#-db DATABSE -u USERNAME -p PASSWORD -size 20

parser.add_argument("-tf", "--textfile", help="Textfiles", type=bool)

args = parser.parse_args()
if(args.textfile == True):
    print(args.textfile)
def sqrt(x):
    if (x < 0):
        return 1j * math.sqrt(abs(x))
    if (x >= 0):
        return math.sqrt(x)

def eff_onsite(kx, ky):
    return par.onsite +2 * (math.cos(kx) + math.cos(ky))

kx = [0 for m in range(0, par.chain_length_x)]
ky = [0 for m in range(0, par.chain_length_y)]
dos = [0 for r in range(0, par.steps)]

for i in range(0, par.chain_length_y):
    if (par.chain_length_y != 1):
        ky[i] = 2 * par.pi * i / par.chain_length_y
    elif (par.chain_length_y == 1):
        ky[i] = par.pi / 2.0

for i in range(0, par.chain_length_x):
    if (par.chain_length_x != 1):
        kx[i] = 2 * par.pi * i / par.chain_length_x
    elif (par.chain_length_x == 1):
        kx[i] = par.pi / 2.0

for i in range(0, par.chain_length_x):
    for j in range(0, par.chain_length_y):
        for r in range(0, par.steps):
            dos[r] += 1 / par.pi * (1 /sqrt(4 * par.hopping ** 2 - (par.energy[r].real - eff_onsite(kx[i], ky[j])) ** 2)) 

fig = plt.figure()
plt.plot([e.real for e in par.energy], 
    [e.real for e in dos], color='blue', label='Real self energy 50 k points')

plt.title('The DOS')
plt.legend(loc='upper left')
plt.xlabel("energy")
plt.ylabel("Self Energy")
plt.show()

"""
def bandstructure(kx, ky, kz):
    return par.onsite + 2 * (math.cos(kx) + math.cos(ky) + math.cos(kz))

num_kz = 80
kx = [0 for m in range(0, par.chain_length_x)]
ky = [0 for m in range(0, par.chain_length_y)]
kz = [2 * par.pi / (num_kz + 1) * x for x in range(1 , num_kz + 1)]
dos = [0 for r in range(0, par.steps)]

n = par.chain_length_x * par.chain_length_y

for i in range(0, par.chain_length_y):
    if (par.chain_length_y != 1):
        ky[i] = 2 * par.pi * i / par.chain_length_y
    elif (par.chain_length_y == 1):
        ky[i] = par.pi / 2.0

for i in range(0, par.chain_length_x):
    if (par.chain_length_x != 1):
        kx[i] = 2 * par.pi * i / par.chain_length_x
    elif (par.chain_length_x == 1):
        kx[i] = par.pi / 2.0

for x in range(0, num_kz):
    for i in range(0, par.chain_length_x):
        for j in range(0, par.chain_length_y):
            for r in range(0, par.steps):
                dos[r] -= 1 / (par.pi * (par.energy[r] - bandstructure(kx[i], ky[j], kz[x]) + .00001j))


fig = plt.figure()
plt.plot([e.real for e in par.energy], 
    [e.imag for e in dos], color='blue', label='Analytical DOS')

plt.title('The DOS')
plt.legend(loc='upper left')
plt.xlabel("energy")
plt.ylabel("Self Energy")
plt.show()
"""