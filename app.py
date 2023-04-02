
import numpy as np
from keras.models import load_model
from numpy import random
import pandas as pd


model = load_model('model.h5')

def transformer(Hydrogen,Oxigen,Nitrogen,Methane,CO,CO2,Ethylene,Ethane,Acethylene,DBDS,Power_factor,Interfacial_V,Dielectric_rigidity,Water_content,Life_expectation):
    data = np.array([[Hydrogen,Oxigen,Nitrogen,Methane,CO,CO2,Ethylene,Ethane,Acethylene,DBDS,Power_factor,Interfacial_V,Dielectric_rigidity,Water_content,Life_expectation]])
    prediction = model.predict(data)
    x = int(prediction[0])
    if x >= 56:
        transformer_index = f'Transformer index is 1 with a health index percentage of {x}'
    elif 29<=x<=55:
        transformer_index = f'Transformer index is 2 with a health index percentage of {x}'
    elif 17<=x<=28:
        transformer_index = f'Transformer index is 3 with a health index percentage of {x}'
    elif 6<=x<=16:
        transformer_index = f'Transformer index is 4 with a health index percentage of {x}'
    elif x<=5:
        transformer_index = f'Transformer index is 5 with a health index percentage of {x}'

    return transformer_index

myArray = np.array([])

def cock():
    myArray = np.array([])
    for n in range(10):
        a = random.randint(12886)
        b = random.randint(21800)
        c = random.randint(60600)
        d = random.randint(7406)
        e = random.randint(520)
        f = random.randint(2480)
        g = random.randint(16684)
        h = random.randint(1450)
        j = random.randint(164)
        k = random.randint(500)/100
        l = random.randint(100)
        z = random.randint(60)
        x = random.randint(70)
        v = random.randint(30)
        transformer(a,b,c,d,d,e,f,g,h,j,k,l,z,x,v)
        myArray = np.append(myArray,x)
        return(myArray)

    
print(cock())



