import matplotlib.pyplot as plt


voltage = [ 0.1 * i for i in range(11)]
current = [0.0 , 0.013188346569872573 , 0.026318557880102133 , 0.0377953170196137 , 0.29201874269749023 , 0.30136112669702503  , 0.3111619416009072 , 0.32078759984715266 , 0.3302141369245465 , 0.33916299309989073 ,0.3453269840937152 ]

voltage.append(1.2)
voltage.append(1.5)
voltage.append(2.0)
current.append(0.525677916826721)
current.append(0.5706109677459779)
current.append(0.6535677916826721)


plt.plot( voltage , current , color='blue' ) 
plt.title("Current vs Voltage")
    #plt.legend(loc='upper right')
plt.xlabel("Voltage")
plt.ylabel("Current")  
plt.show()