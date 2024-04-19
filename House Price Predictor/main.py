from tkinter import *
from tkinter import font
from tkinter.font import BOLD, ITALIC
from tkinter import ttk
from joblib import dump, load
import numpy as np
import tkinter.messagebox as tm


model = load('model.joblib')

def predictator():
    crim = float(crimvalue.get())
    zn = float(znvalue.get())
    indus = float(indusvalue.get())
    chas = float(chasvalue.get())
    nox = float(noxvalue.get())
    rm = float(rmvalue.get())
    age= float(agevalue.get())
    dis = float(disvalue.get())
    rad = float(radvalue.get())
    tax = float(taxvalue.get())
    ptratio = float(ptratiovalue.get())
    b = float(bvalue.get())
    lstat = float(lstatvalue.get())

    features = np.array([[crim, zn, indus, chas, nox, rm, age, dis , rad, tax, ptratio, b, lstat]])
    print(model.predict(features))
    tm.showinfo("PREDICTED PRICE",f"The Predicted Price is ${1000*(model.predict(features)[0])}")

root = Tk()

root.maxsize(1080, 540)
root.minsize(1080, 540)
root.title("Mini Project")
root.config(bg='#222222')

#Create a main frame
main_frame = Frame(root)
main_frame.pack(fill=BOTH, expand=1)
# #Create a canvas
my_canvas = Canvas(main_frame)
my_canvas.pack(side=LEFT, expand=1, fill=BOTH)
# #Add a scrollbar to the canvas
my_scrollbar = ttk.Scrollbar(main_frame, orient=VERTICAL, command=my_canvas.yview,)
my_scrollbar.pack(side=RIGHT, fill=Y)
# #Configure the canvas
my_canvas.configure(yscrollcommand=my_scrollbar.set)
my_canvas.bind('<Configure>', lambda e: my_canvas.configure(scrollregion= my_canvas.bbox("all")))
# #Create another frame inside the canvas
second_frame = Frame(my_canvas)
second_frame.config(bg='#222222')
# # Add that new frame to a window in the canvas
my_canvas.create_window((0, 0), window= second_frame, anchor= NW)
my_canvas.config(bg='#222222')


root.config(bg='#222222')
#Lables

def createLabels(text, labelName):
    labelName = Label(second_frame,text= f"{text}",foreground='#EEEEEE', background="#222222",font=('Bahnschrift', 13), )
    labelName.pack(anchor=NW, ipadx=15, ipady=5)

def createEntry(entryName):
    entryName = Entry(second_frame, textvariable=StringVar(), background='#222222', width= 172, foreground='#EEEEEE')
    entryName.pack(anchor=NW, padx=15, ipady=5)
    return(entryName)

l1 = Label(second_frame,text="House Price Predictor",foreground='#EEEEEE',bg='#222222',font=('comicsansms',22,BOLD))
l1.pack(side=TOP, ipady=22)

l2 = Label(second_frame,text="Enter The Features Of House:",foreground='#EEEEEE', background="#222222", font=('Bahnschrift', 16))
l2.pack(anchor=NW, ipadx=15, ipady= 10)

#CRIM:: value = 0 to 100
createLabels('Per capita crime rate by town:', 'l3')
crimvalue = createEntry('crimvalue')

#ZN: value = 0 to 100
createLabels('Proportion of residential land zoned for lots over 25,000 sq.ft:', 'l4')
znvalue = createEntry('znvalue')

#INDUS: value => 1 to 30
createLabels('INDUS (proportion of non-retail business acres per town:', 'l5')
indusvalue= createEntry('indusvalue')

#CHAS: value = 0 or 1
createLabels('Charles River dummy variable (= 1 if tract bounds river; 0 otherwise:', 'l6')
chasvalue = createEntry('chasvalue')

#NOX: value = 0 to 1
createLabels('Nitric oxides concentration (parts per 10 million):', 'l7')
noxvalue = createEntry('noxvalue')

#RM: value = 5 to 10
createLabels('Average number of rooms per dwelling:', 'l8')
rmvalue = createEntry('rmvalue')

#AGE: value = 0 to 100
createLabels('Proportion of owner-occupied units built prior to 1940:', 'l9')
agevalue = createEntry('agevalue')

#DIS: value = 1 to 10
createLabels('Weighted distances to five Boston employment centres:', 'l10')
disvalue = createEntry('disvalue')

#RAD: value = 1 to 25
createLabels('Index of accessibility to radial highways:', 'l11')
radvalue = createEntry('radvalue')

#TAX: value = 200 to 800
createLabels('Full-value property-tax rate per $10,000:', 'l12')
taxvalue = createEntry('taxvalue')

#PTRATIO: value = 15 to 21
createLabels('Pupil-teacher ratio by town:', 'l13')
ptratiovalue = createEntry('ptratiovalue')

#B: value = 1 to 400
createLabels('1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town:', 'l14')
bvalue = createEntry('bvalue')

#LSTAT: value = 1 to 20
createLabels('Lower status of the population:', 'l15')
lstatvalue = createEntry('lstatvalue')

bt = Button(second_frame, text="PREDICT", command=predictator)
bt.pack(side= BOTTOM,anchor=N, padx=15, ipady=5, pady= 8)

root.mainloop()  
