import numpy as np
import matplotlib.pyplot as plt

def ricker(f, length=0.128, dt=0.001):
    t = np.arange(-length/2, (length-dt)/2, dt)
    y = (1.0 - 2.0*(np.pi**2)*(f**2)*(t**2)) * np.exp(-(np.pi**2)*(f**2)*(t**2))
    return t, y
 
f = 25 # A low wavelength of 25 Hz
t, w = ricker(f)

print(t)
print('****')
print(w)

plt.plot(w)
plt.show()


t0 = self.t0 or 1 / self.f0
a = self.a or 1
r = (np.pi * self.f0 * (self.time_values - t0))
return a * (1-2.*r**2)*np.exp(-r**2)
