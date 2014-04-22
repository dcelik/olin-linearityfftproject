# -*- coding: utf-8 -*-
"""
Created on Fri Apr 18 14:00:00 2014

@author: dcelik
"""

from numpy import fft
import numpy as np 
import matplotlib.pyplot as plt 

def makeplot(func):
    n = 128 # Number of data points 
    dt = 5.0 # Sampling period (in meters) 
    x = dt*np.arange(0,n) # x coordinate 
    w1 = 100.0 # wavelength (meters) 
    w2 = 20.0 # wavelength (meters) 
    fx = np.sin(2*np.pi*x/w1) + 2*np.cos(2*np.pi*x/w2) # signal
    fx = np.cos(2*x)
    
    
    z, ax = plt.subplots(3,1)
    ax[0].plot(x,fx)
    
    
    
    Fk = fft.fft(fx)/n # Fourier coefficients (divided by n) 
    nu = fft.fftfreq(n,dt) # Natural frequencies 
    
    print Fk
    print nu    
    
    Fk = fft.fftshift(Fk) # Shift zero freq to center 
    nu = fft.fftshift(nu) # Shift zero freq to center 
    
    print Fk
    print nu

    f, ax = plt.subplots(3,1,sharex=True) 
    ax[0].plot(nu, np.real(Fk)) # Plot Cosine terms 
    ax[0].set_ylabel(r'$Re[F_k]$', size = 'x-large') 
    ax[1].plot(nu, np.imag(Fk)) # Plot Sine terms 
    ax[1].set_ylabel(r'$Im[F_k]$', size = 'x-large') 
    ax[2].plot(nu, np.absolute(Fk)**2) # Plot spectral power 
    ax[2].set_ylabel(r'$\vert F_k \vert ^2$', size = 'x-large') 
    ax[2].set_xlabel(r'$\widetilde{\nu}$', size = 'x-large') 
    plt.show()
makeplot("x")