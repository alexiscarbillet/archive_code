import sys
import os
from tkinter import *

window=Tk()

window.title("Running Python Script")
window.geometry('670x200')
window.configure(bg="black")

def run_calc():
    os.system('python Calculator/calculator.py')

def run_sna():
    os.system('python Snake/snake.py')

def run_tet():
    os.system('python Tetris/tetris.py')

def run_bre():
    os.system('python Breakout/breakout.py')


label_1 = Label(window,text = "Games menu", bg = "black", fg = "white", font = ("Arial", 20))
label_1.grid(columnspan=3,  pady= 30 , padx = 20 , ipadx = 10 , ipady = 10)

btn = Button(window, text="Calculator", bg="black", fg="white",command=run_calc, width=30)
btn.grid(column=0, row=1)

btn2 = Button(window, text="Snake", bg="black", fg="white",command=run_sna, width=30)
btn2.grid(column=1, row=1)

btn3 = Button(window, text="Tetris", bg="black", fg="white",command=run_tet, width=30)
btn3.grid(column=2, row=1)

btn4 = Button(window, text="Breakout", bg="black", fg="white",command=run_bre, width=30)
btn4.grid(column=0, row=2)

window.mainloop()