# -*- coding: utf-8 -*-
"""
Created on Fri Apr 18 14:00:00 2014

@author: dcelik
"""

import numpy as np#import the math tools
import pylab as pyl
import scipy as spy
import time

Fs = 4096.0#sampling rate of function, frequency CANNOT EXCEED THIS VALUE: will lead to memory leakage

def parsestring(func):
    if len(func)>=6:#make sure minimum length is reached
        func = func.lower()#make lower case to prevent mismatch based on case
        funcl = func.split('+')#split each trig function into distinct elements
        l = []#initiliaze list for function value
        for i in funcl:#loop through distinct trig functions
            l.append(parsefunc(i))#evaluate and append each trig function value to list
    else:
        print "your function is too short!"
        return None
    dfunc = evaluatefunc(l)#evaluate the transform of the discrete points created by the input function
    makeplot(dfunc)#plot the relevant transform info
    

def parsefunc(func):
    amp = str(1)#set up parameter defaults
    trig = ""
    freq = str(1)
    pi = False
    funcl = func[:len(func)-1].split("(")#split function into trig function and inside
    if funcl[0].find("*")!=-1:#check if multiplication is done before trig ie. amp>1
        amp = funcl[0][0:funcl[0].index("*")]
        trig = funcl[0][funcl[0].find(amp)+len(amp)+1:]
    elif len(funcl[0])==3:
        trig = funcl[0]
    else:
        print "that is not allowed."
        return
        
    if funcl[1].find("*")!=-1:#check if multiplication is done inside trig ie. pi or f present
        freql = funcl[1].split("*")#split at multi
        if len(freql)==2:#if only two then either pi or f
            if freql[0]=="pi":
                pi = True
            else:
                freq = float(freql[0])#set other value to f
                pi = False
        elif len(freql)==3:#if three then f and pi are present
            freq = float(freql[0])
            pi = True
    return [float(amp),trig,freq,pi]#pass parameters back to main string parsing
 
 
 
def evaluatefunc(func):
    t = spy.arange(0,1,1/Fs)#creating x range
    fl = []#initialize list of trig function values
    for i in func:#loop through each distinct trig function in func
        if i[1]=='sin':#check if sin is present
            if i[3]:#check if pi needs to be inserted
                fl.append(i[0]*np.sin(i[2]*np.pi*t))
            else:
                fl.append(i[0]*np.sin(i[2]*t))
        elif i[1]=='cos':
            if i[3]:
                fl.append(i[0]*np.cos(i[2]*np.pi*t))
            else:
                fl.append(i[0]*np.cos(i[2]*t))
    return sum(fl)#return the sum off all distinct trig functions for each time point

def makeplot(func):
    def plotabsnp(y,Fs):
        frq = (spy.arange(len(y))/(len(y)/Fs))[0:len(y)/2]#create range of frequency values        
        time_start = time.clock()#start timing
        Y = (spy.fft(y)/len(y))[0:len(y)/2]#do fourier transform on y then take only positive half
        time_elapsed = (time.clock() - time_start)#end timing
        
        pyl.plot(frq,abs(Y),'r')#plot absolute value of transform
        pyl.xlabel('Freq (Hz)')#labeling
        pyl.ylabel('|Y(freq)|^2')
        return time_elapsed#return the timing for the transform
        
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
        f = np.asarray(f, dtype=float)#turn list into numpy array
        N = f.shape[0]#set length of array to N
        n = np.arange(N)#make list n with time values
        k = n.reshape((N,1))#transpose n into k
        M = np.exp(-2j*np.pi*k*n/N)#create fourier transform matrix
        return np.dot(M, f)# dot matrix with original data to transform
        
    def fftfast(x):
        x = np.asarray(x, dtype=float)#turn list into numpy array
        N = x.shape[0]#set length of array to N

        if N % 2 > 0:#check to make sure we can use algorithm
            raise ValueError("size of x must be a power of 2")# raise error and break to ask for new sample rate
        elif N <= 32:#cutoff where slow is faster than using symmetry
            return fftslow(x)#use slow to save time
        else:#recurse!
            X_even = fftfast(x[::2])#calculate the even part of x
            X_odd = fftfast(x[1::2])#calculate the odd part of x
            factor = np.exp(-2j * np.pi * np.arange(N) / N)# calculate factor that makes even part into odd
            return np.concatenate([X_even + factor[:N / 2] * X_odd,
                                   X_even + factor[N / 2:] * X_odd])#concatenate the odd and even to get the full fourier transform
        
    t = spy.arange(0,1,1/Fs)#create x values (frequency), 128 samples to space 1 second of signal makes graphs look nice
    
    pyl.subplots_adjust(hspace=.5)
    pyl.subplot(4,1,1)#original plot
    pyl.plot(t,func)
    pyl.xlabel('Time')
    pyl.ylabel('Amplitude')
    
    pyl.subplot(4,3,4)#imaginary part
    tnp1 = plotimgnp(func,Fs)
    pyl.title('Numpy FFT')
    
    pyl.subplot(4,3,7)#real part
    tnp2 = plotrealnp(func,Fs)
    
    pyl.subplot(4,3,10)#power spectrum
    tnp3 = plotabsnp(func,Fs)

    print (tnp1+tnp2+tnp3)/3.0#time for numpy
    
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
    
    pyl.show()#SHOW ME DA MONEY. or plots. im a comment. not a policeman.