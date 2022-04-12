import scipy.linalg as la
import matplotlib.pyplot as plt


energy_ks = [-456.2254, -460.2777, -462.8240, -464.1386, -465.4434, -466.6984, -467.8612, -468.9034, -469.7410, -470.2980, -470.4277,  -470.4324, -470.4359, -470.4358, -470.4343, -470.4283,  -470.4111, -469.8322 ,  -468.2214 , -465.2164, -460.1002, -451.6753]
lattice_const= [ 6.4, 6.0, 5.6, 5.4, 5.2, 5.0, 4.8, 4.6, 4.4, 4.2, 4.1, 4.09, 4.07, 4.06, 4.05, 4.03, 4, 3.8, 3.6, 3.4, 3.2, 3.0]

fig = plt.figure()
plt.plot(lattice_const, 
    energy_ks, color='blue')
plt.title('Energy vs lattice constant')
#plt.legend(loc='upper right')
plt.xlabel("lattice const(Ang)")
plt.ylabel("energy(eV)")
plt.show()

-470.4343