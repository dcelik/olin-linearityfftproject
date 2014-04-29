# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 14:02:08 2014

@author: dcelik
"""


import Tkinter as tk
import fft


class gui(tk.Tk):
    def __init__(self,parent): 
        tk.Tk.__init__(self,parent)
        self.parent = parent
        self.about_messageprog = "AN APPLICATION OF THE FAST FOURIER TRANSFORM FOR DIFFERENT SIGNAL INPUTS."
        self.about_messagecreat = "CREATED BY DENIZ CELIK AND JACOB RIEDEL. BOTH STUDENTS ARE FRESHMAN AT THE FRANKLIN W. OLIN COLLEGE OF ENGINEERING IN NEEDHAM, MA."              
        self.initialize()
        
    def aboutprog(self):
        popup = tk.Toplevel()
        popup.title("About this program")
        
        msg = tk.Message(popup,text=self.about_messageprog)
        msg.pack()
        
        button = tk.Button(popup,text="Close", command = popup.destroy)
        button.pack()
        
    def aboutcreat(self):
        popup = tk.Toplevel()
        popup.title("About the creators")
        
        msg = tk.Message(popup,text=self.about_messagecreat)
        msg.pack()
        
        button = tk.Button(popup,text="Close", command = popup.destroy)
        button.pack()

    def makemenu(self):
        #creating submenus for each item
        self.aboutsubMenu = tk.Menu(self.menuBar)
        
        #about submenu commands
        self.menuBar.add_cascade(label='About',menu = self.aboutsubMenu)
        self.menuBar.add_command(label='Quit', command=self.quit)
        
        #about submenu commands
        self.aboutsubMenu.add_command(label='About the Program', command=self.aboutprog)
        self.aboutsubMenu.add_command(label='About the Creators',command=self.aboutcreat)
            
    def initialize(self):
        #initialize grid
        self.grid()
        
        #set minimum size500 and prevent resizing
        self.minsize(735,150)
        self.resizable(False,False)        
        
        #top level gui creation
        top = self.winfo_toplevel()
        self.menuBar = tk.Menu(top)
        top['menu'] = self.menuBar
        
        self.makemenu()    
    
        self.instruc1 = tk.Label(self.parent,font=("Comic Sans",20),text = "Input your function in the form: \'A*Trigfunc(B*pi*t)+...\'.")    
        self.instruc1.grid(column = 0, row=0, columnspan = 2,sticky = "NEWS")
        self.instruc2 = tk.Label(self.parent,font=("Comic Sans",20),text = "Click the enter button to produce the FFT graphs")    
        self.instruc2.grid(column = 0, row=2, columnspan = 2,sticky = "NEWS")
        self.funcinp = tk.Entry(self.parent)
        self.funcinp.grid(column = 0, row=1, sticky = "NEWS")
        
        self.enterbut = tk.Button(self.parent,font=("Comic Sans",20),text = "Enter", command = lambda: fft.parsestring(self.funcinp.get()))
        self.enterbut.grid(column = 1, row=1, sticky = "NEWS")        
        
        #update and geometry
        self.update()
        self.geometry(self.geometry())     
        
    
        
if __name__ == "__main__":
    app = gui(None)
    app.title('Linearity 1: Fast Fourier Transform')
    app.mainloop()