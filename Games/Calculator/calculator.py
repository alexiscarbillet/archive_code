from tkinter import *
import math

formule = "" 
 
def click(num): 
 
    global formule 
    formule = formule + str(num) 
    equation.set(formule) 
 
def equalclick(): 
    try: 
        global formule 
 
        result = str(eval(formule)) 
        equation.set(result) 
        formule = result
 
    except: 
        equation.set(" error ") 
        formule = "" 
 
def effacer(): 
    global formule 
    formule = "" 
    equation.set("") 

def expo():
    global formule 
    equalclick()
    x = str(math.exp(float(formule)))
    formule = ""
    click(x)

def log():
    global formule 
    equalclick()
    x = str(math.log10(float(formule)))
    formule = ""
    click(x)

def ln():
    global formule 
    equalclick()
    x = str(math.log2(float(formule)))
    formule = ""
    click(x)

def sin():
    global formule 
    equalclick()
    x = str(math.sin(float(formule)))
    formule = ""
    click(x)

def cos():
    global formule 
    equalclick()
    x = str(math.cos(float(formule)))
    formule = ""
    click(x)

def tan():
    global formule 
    equalclick()
    x = str(math.tan(float(formule)))
    formule = ""
    click(x)

def fac():
    global formule 
    equalclick()
    x = str(math.factorial(int(formule)))
    formule = ""
    click(x)

def root():
    global formule 
    equalclick()
    x = str(math.sqrt(float(formule)))
    formule = ""
    click(x)

def inv():
    global formule 
    equalclick()
    x = str(1/float(formule))
    formule = ""
    click(x)
 
if __name__ == "__main__": 
    master = Tk() 
    master.title("Calculatrice") 
    master.geometry("400x415") 
    master.config(bg='#FFFFAA')
    equation = StringVar() 
    formule_field = Entry(master, textvariable=equation) 
    formule_field.grid(columnspan=5,  pady= 30 , padx = 20 , ipadx = 100 , ipady = 10)
    btn_1 = Button(master, text=' 1 ', command=lambda: click(1), height=2, width=10) 
    btn_1.grid(row=2, column=0) 
 
    btn_2 = Button(master, text=' 2 ', command=lambda: click(2), height=2, width=10) 
    btn_2.grid(row=2, column=1) 
 
    btn_3 = Button(master, text=' 3 ', command=lambda: click(3), height=2, width=10) 
    btn_3.grid(row=2, column=2) 

    btn_cyan = Button(master, text=' cyan ', command=lambda: master.config(bg='#AAFFFF'), height=2, width=10) 
    btn_cyan.grid(row=2, column=4) 

    btn_4 = Button(master, text=' 4 ', command=lambda: click(4), height=2, width=10) 
    btn_4.grid(row=3, column=0) 
 
    btn_5 = Button(master, text=' 5 ', command=lambda: click(5), height=2, width=10) 
    btn_5.grid(row=3, column=1) 
 
    btn_6 = Button(master, text=' 6 ', command=lambda: click(6), height=2, width=10) 
    btn_6.grid(row=3, column=2) 

    btn_root = Button(master, text=' sqrt ', command=root, height=2, width=10) 
    btn_root.grid(row=3, column=4) 
 
    btn_7 = Button(master, text=' 7 ', command=lambda: click(7), height=2, width=10) 
    btn_7.grid(row=4, column=0) 
 
    btn_8 = Button(master, text=' 8 ', command=lambda: click(8), height=2, width=10) 
    btn_8.grid(row=4, column=1) 
 
    btn_9 = Button(master, text=' 9 ', command=lambda: click(9), height=2, width=10) 
    btn_9.grid(row=4, column=2) 

    ln = Button(master, text=' ln ', command=ln, height=2, width=10) 
    ln.grid(row=4, column=4)
 
    btn_0 = Button(master, text=' 0 ', command=lambda: click(0), height=2, width=10) 
    btn_0.grid(row=5, column=0) 
 
    plus = Button(master, text=' + ', command=lambda: click("+"), height=2, width=10) 
    plus.grid(row=2, column=3) 
 
    minus = Button(master, text=' - ', command=lambda: click("-"), height=2, width=10) 
    minus.grid(row=3, column=3) 
 
    multiply = Button(master, text=' * ', command=lambda: click("*"), height=2, width=10) 
    multiply.grid(row=4, column=3) 
 
    divide = Button(master, text=' / ', command=lambda: click("/"), height=2, width=10) 
    divide.grid(row=5, column=2) 
 
    equal = Button(master, text=' = ', command=equalclick, height=2, width=10) 
    equal.grid(row=6, column=3) 
 
    effacer = Button(master, text='clear', command=effacer, height=2, width=10) 
    effacer.grid(row=6, column=0) 

    fac = Button(master, text=' x! ', command=fac, height=2, width=10) 
    fac.grid(row=6, column=4) 
 
    Decimal= Button(master, text='.', command=lambda: click('.'), height=2, width=10) 
    Decimal.grid(row=5, column=1) 
    
    percent= Button(master, text='%', command=lambda: click('%'), height=2, width=10) 
    percent.grid(row=6, column=1) 
    
    power= Button(master, text='**', command=lambda: click('**'), height=2, width=10) 
    power.grid(row=6, column=2)   

    pi= Button(master, text='pi', command=lambda: click(str(math.pi)), height=2, width=10) 
    pi.grid(row=5, column=3)   

    inv = Button(master, text='1/x', command=inv , height=2, width=10) 
    inv.grid(row=5, column=4)   

    btn_green = Button(master, text=' green ', command=lambda: master.config(bg='#AAFFAA'), height=2, width=10)
    btn_green.grid(row=7, column=0)

    btn_blue = Button(master, text=' blue ', command=lambda: master.config(bg='#AAAAFF'), height=2, width=10)
    btn_blue.grid(row=7, column=1)

    btn_red = Button(master, text=' red ', command=lambda: master.config(bg='#FFAAAA'), height=2, width=10)
    btn_red.grid(row=7, column=2)

    btn_yellow = Button(master, text=' yellow ', command=lambda: master.config(bg='#FFFFAA'), height=2, width=10)
    btn_yellow.grid(row=7, column=3)

    btn_fushia = Button(master, text=' fushia ', command=lambda: master.config(bg='#FFAAFF'), height=2, width=10)
    btn_fushia.grid(row=7, column=4)

    btn_log = Button(master, text=' log ', command=log, height=2, width=10)
    btn_log.grid(row=8, column=0)

    btn_exp = Button(master, text=' exp ', command=expo, height=2, width=10)
    btn_exp.grid(row=8, column=1)

    btn_cos = Button(master, text=' cos ', command=cos, height=2, width=10)
    btn_cos.grid(row=8, column=2)
     
    btn_sin = Button(master, text=' sin ', command=sin, height=2, width=10)
    btn_sin.grid(row=8, column=3)

    btn_tan = Button(master, text=' tan ', command=tan, height=2, width=10)
    btn_tan.grid(row=8, column=4)
    
    master.mainloop()