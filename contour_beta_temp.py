import matplotlib.pyplot as plt
from numpy import loadtxt
import numpy as np
import numpy
import matplotlib.pyplot as plt
from astropy.io import fits
#def plotbet_temp():
from scipy.stats import gaussian_kde
data = loadtxt("Results/data.dat")
temp = data[:,2]
beta = data[:,4]
x=temp
y=beta
xy = np.vstack([x,y])
z = gaussian_kde(xy)(xy)

fig, ax = plt.subplots()
plt.scatter(x, y, c=z, s=1, edgecolor='')
plt.xlim(12, 22)
plt.ylim(1.2,2.2)
plt.xlabel(r'$Temperature$')
plt.ylabel(r'$Beta$')
plt.grid(True)
plt.savefig('Results/Beta_Temp_corr.png')
plt.show()

