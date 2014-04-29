# -*- coding: utf-8 -*-
"""
Created on Fri Apr 18 14:00:00 2014

@author: dcelik
"""

import numpy as np
import pylab as pyl
import scipy as spy
import time

Fs = 128.0

DATA_NUM = 1000
def parsestring(func):
    if len(func)>=6:
        func = func.lower()
        funcl = func.split('+')
        l = []
        for i in funcl:
            l.append(parsefunc(i))
    else:
        print "your function is too short!"
        return None
    dfunc = evaluatefunc(l)
    makeplot(dfunc)
    

def parsefunc(func):
    amp = str(1)
    trig = ""
    freq = str(1)
    pi = False
    funcl = func[:len(func)-1].split("(")
    if funcl[0].find("*")!=-1:
        amp = funcl[0][0:funcl[0].index("*")]
        trig = funcl[0][funcl[0].find(amp)+len(amp)+1:]
    elif len(funcl[0])==3:
        trig = funcl[0]
    else:
        print "that is not allowed."
        return
        
    if funcl[1].find("*")!=-1:
        freql = funcl[1].split("*")
        if len(freql)==2:
            if freql[0]=="pi":
                pi = True
            else:
                freq = float(freql[0])
                pi = False
        elif len(freql)==3:
            freq = float(freql[0])
            pi = True
    return [float(amp),trig,freq,pi]
 
 
 
def evaluatefunc(func):
    t = spy.arange(0,Fs/128,(Fs/128)/Fs)
    fl = []
    for i in func:
        if i[1]=='sin':
            if i[3]:
                fl.append(i[0]*np.sin(i[2]*np.pi*t))
            else:
                fl.append(i[0]*np.sin(i[2]*t))
        elif i[1]=='cos':
            if i[3]:
                fl.append(i[0]*np.cos(i[2]*np.pi*t))
            else:
                fl.append(i[0]*np.cos(i[2]*t))
    return sum(fl)
    
def makeplot(func):
    def plotabsnp(y,Fs):
        frq = (spy.arange(len(y))/(len(y)/Fs))[0:len(y)/2]        
        time_start = time.clock()
        Y = (spy.fft(y)/len(y))[0:len(y)/2]
        time_elapsed = (time.clock() - time_start)
        
        pyl.plot(frq,abs(Y),'r')
        pyl.xlabel('Freq (Hz)')
        pyl.ylabel('|Y(freq)|^2')
        return time_elapsed
        
    def plotimgnp(y,Fs):
        frq = (spy.arange(len(y))/(len(y)/Fs))[0:len(y)/2]        
        time_start = time.clock()
        Y = (spy.fft(y)/len(y))[0:len(y)/2]
        time_elapsed = (time.clock() - time_start)
        Yim = (np.imag(Y))[0:len(y)/2]
     
        pyl.plot(frq,abs(Yim),'r')
        pyl.xlabel('Freq (Hz)')
        pyl.ylabel('Im[Y] (freq)')
        return time_elapsed
    
    def plotrealnp(y,Fs):
        frq = (spy.arange(len(y))/(len(y)/Fs))[0:len(y)/2]   
        time_start = time.clock()
        Y = (spy.fft(y)/len(y))[0:len(y)/2]
        time_elapsed = (time.clock() - time_start)
        Yreal = (np.real(Y))[0:len(y)/2]
        
        pyl.plot(frq,abs(Yreal),'r')
        pyl.xlabel('Freq (Hz)')
        pyl.ylabel('Re[Y] (freq)')
        return time_elapsed
        
        
        
    def plotabsslow(y,Fs):
        frq = (spy.arange(len(y))/(len(y)/Fs))[0:len(y)/2]        
        time_start = time.clock()
        Y = (fftslow(y)/len(y))[0:len(y)/2]
        time_elapsed = (time.clock() - time_start)
        
        pyl.plot(frq,abs(Y),'r')
        pyl.xlabel('Freq (Hz)')
        pyl.ylabel('|Y(freq)|^2')
        return time_elapsed
        
    def plotimgslow(y,Fs):
        frq = (spy.arange(len(y))/(len(y)/Fs))[0:len(y)/2]        
        time_start = time.clock()
        Y = (fftslow(y)/len(y))[0:len(y)/2]
        time_elapsed = (time.clock() - time_start)
        Yim = (np.imag(Y))[0:len(y)/2]
     
        pyl.plot(frq,abs(Yim),'r')
        pyl.xlabel('Freq (Hz)')
        pyl.ylabel('Im[Y] (freq)')
        return time_elapsed
    
    def plotrealslow(y,Fs):
        frq = (spy.arange(len(y))/(len(y)/Fs))[0:len(y)/2]   
        time_start = time.clock()
        Y = (fftslow(y)/len(y))[0:len(y)/2]
        time_elapsed = (time.clock() - time_start)
        Yreal = (np.real(Y))[0:len(y)/2]
        
        pyl.plot(frq,abs(Yreal),'r')
        pyl.xlabel('Freq (Hz)')
        pyl.ylabel('Re[Y] (freq)')
        return time_elapsed
        
        
        
    def plotabsfast(y,Fs):
        frq = (spy.arange(len(y))/(len(y)/Fs))[0:len(y)/2]        
        time_start = time.clock()
        Y = (fftfast(y)/len(y))[0:len(y)/2]
        time_elapsed = (time.clock() - time_start)
        
        pyl.plot(frq,abs(Y),'r')
        pyl.xlabel('Freq (Hz)')
        pyl.ylabel('|Y(freq)|^2')
        return time_elapsed
        
    def plotimgfast(y,Fs):
        frq = (spy.arange(len(y))/(len(y)/Fs))[0:len(y)/2]        
        time_start = time.clock()
        Y = (fftfast(y)/len(y))[0:len(y)/2]
        time_elapsed = (time.clock() - time_start)
        Yim = (np.imag(Y))[0:len(y)/2]
     
        pyl.plot(frq,abs(Yim),'r')
        pyl.xlabel('Freq (Hz)')
        pyl.ylabel('Im[Y] (freq)')
        return time_elapsed
    
    def plotrealfast(y,Fs):
        frq = (spy.arange(len(y))/(len(y)/Fs))[0:len(y)/2]   
        time_start = time.clock()
        Y = (fftfast(y)/len(y))[0:len(y)/2]
        time_elapsed = (time.clock() - time_start)
        Yreal = (np.real(Y))[0:len(y)/2]
        
        pyl.plot(frq,abs(Yreal),'r')
        pyl.xlabel('Freq (Hz)')
        pyl.ylabel('Re[Y] (freq)')
        return time_elapsed
        
    def fftslow(f):
        f = np.asarray(f, dtype=float)
        N = f.shape[0]
        n = np.arange(N)
        k = n.reshape((N, 1))
        M = np.exp(-2j * np.pi * k * n / N)
        return np.dot(M, f)
        
    def fftfast(x):
        """A recursive implementation of the 1D Cooley-Tukey FFT"""
        x = np.asarray(x, dtype=float)
        N = x.shape[0]
        
        if N % 2 > 0:
            raise ValueError("size of x must be a power of 2")
        elif N <= 32:  # this cutoff should be optimized
            return fftslow(x)
        else:
            X_even = fftfast(x[::2])
            X_odd = fftfast(x[1::2])
            factor = np.exp(-2j * np.pi * np.arange(N) / N)
            return np.concatenate([X_even + factor[:N / 2] * X_odd,
                                   X_even + factor[N / 2:] * X_odd])
        
    t = spy.arange(0,Fs/128,(Fs/128)/Fs)
    
    pyl.subplots_adjust(hspace=.5)
    pyl.subplot(4,1,1)
    pyl.plot(t,func)
    pyl.xlabel('Time')
    pyl.ylabel('Amplitude')
    
    pyl.subplot(4,3,4)
    tnp1 = plotimgnp(func,Fs)
    pyl.title('Numpy FFT')
    
    pyl.subplot(4,3,7)
    tnp2 = plotrealnp(func,Fs)
    
    pyl.subplot(4,3,10)
    tnp3 = plotabsnp(func,Fs)

    print (tnp1+tnp2+tnp3)/3.0
    
    pyl.subplot(4,3,5)
    ts1 = plotimgslow(func,Fs)
    pyl.title('Slow FFT')
    
    pyl.subplot(4,3,8)
    ts2 = plotrealslow(func,Fs)
    
    pyl.subplot(4,3,11)
    ts3 = plotabsslow(func,Fs)
    
    print (ts1+ts2+ts3)/3.0
    
    pyl.subplot(4,3,6)
    tf1 = plotimgfast(func,Fs)
    pyl.title('Fast FFT')
    
    pyl.subplot(4,3,9)
    tf2 = plotrealfast(func,Fs)
    
    pyl.subplot(4,3,12)
    tf3 = plotabsfast(func,Fs)
    
    print (tf1+tf2+tf3)/3.0
    
    pyl.show()