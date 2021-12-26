import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('FP_mushroom.dat.csv')
data = pd.DataFrame(data)

plt.figure(figsize=(7, 5))
x = range(len(data.iloc[:,0]))
plt.plot(x, data.iloc[:,1])
plt.xticks(x, data.iloc[:,0])
plt.xlabel('Threshold')
plt.ylabel('Runtime(ms)')
plt.show()
plt.savefig("Fp_mushroom_runtime.png")

plt.figure(figsize=(6.8, 4.2))
x2 = range(len(data.iloc[:,0]))
plt.plot(x2, data.iloc[:,2])
plt.xticks(x2, data.iloc[:,0])
plt.xlabel('Threshold')
plt.ylabel('Memory(mb)')
plt.show()
plt.savefig("fp_mushroom_memory.png")