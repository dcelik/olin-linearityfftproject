# -*- coding: utf-8 -*-
"""
Created on Fri Apr 18 14:00:00 2014

@author: dcelik
"""

from numpy import fft
import numpy as np 
import matplotlib.pyplot as plt 

def parsestring(func):
    if len(func)>=6:
        func = func.lower()
        funcl = func.split('+')
        l = ""
        for i in funcl:
            l+=parsefunc(i)+"+"
            return l[:len(l)-1]
    else:
        print "yo func is too short bitch"

def parsefunc(func):
    amp = str(1)
    trig = ""
    freq = str(1)
    funcl = func[:len(func)-1].split("(")
    if funcl[0].find("*")!=-1:
        amp = funcl[0][0:funcl[0].index("*")]
        trig = funcl[0][funcl[0].find(amp)+len(amp)+1:]
    elif len(funcl[0])==3:
        trig = funcl[0]
    else:
        print "that is not allowed. bitch."
        return
    
    if funcl[1].find("*")!=-1:
        freql = funcl[1].split("*")
        if len(freql)==2:
            if freql[0]=="pi":
                freq = "np.pi"
            else:
                freq = freql[0]
        elif len(freql)==3:
            freq = freql[0]+"*"+"np.pi"
    return amp + "*" + "np."+trig+"("+freq+"*"+"x"+")"
    
    
#parsestring("2*Sin(0.002*pi*t)+2*Cos(0.1*pi*t)")

#def makeplot(func):
#    n = 128 # Number of data points 
#    dt = 5.0 # Sampling period (in meters) 
#    x = dt*np.arange(0,n) # x coordinate 
#    w1 = 100.0 # wavelength (meters) 
#    w2 = 20.0 # wavelength (meters) 
#    fx = np.sin(2*np.pi*x/w1) + 2*np.cos(2*np.pi*x/w2) # signal
#    fx = np.cos(2*x)
#    
#    
#    z, ax = plt.subplots(3,1)
#    ax[0].plot(x,fx)
#    
#    
#    
#    Fk = fft.fft(fx)/n # Fourier coefficients (divided by n) 
#    nu = fft.fftfreq(n,dt) # Natural frequencies 
#    
#    print Fk
#    print nu    
#    
#    Fk = fft.fftshift(Fk) # Shift zero freq to center 
#    nu = fft.fftshift(nu) # Shift zero freq to center 
#    
#    print Fk
#    print nu
#
#    f, ax = plt.subplots(3,1,sharex=True) 
#    ax[0].plot(nu, np.real(Fk)) # Plot Cosine terms 
#    ax[0].set_ylabel(r'$Re[F_k]$', size = 'x-large') 
#    ax[1].plot(nu, np.imag(Fk)) # Plot Sine terms 
#    ax[1].set_ylabel(r'$Im[F_k]$', size = 'x-large') 
#    ax[2].plot(nu, np.absolute(Fk)**2) # Plot spectral power 
#    ax[2].set_ylabel(r'$\vert F_k \vert ^2$', size = 'x-large') 
#    ax[2].set_xlabel(r'$\widetilde{\nu}$', size = 'x-large') 
#    plt.show()
#makeplot("x")