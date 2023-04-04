import numpy as np
from keras.models import load_model
from numpy import random
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import datetime as dt
import math
from matplotlib import use as use_agg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import PySimpleGUI as sg

model = load_model('model.h5')

def transformer(Водород,Кислород,Азот,Метан,CO,CO2,Этилен,Этан,Ацетилен,Дибензилсульфид,Напряжение,Поверхностное_Н,Диэлектрическая_жесткость,Содержание_Воды,Срок_службы):
    data = np.array([[Водород,Кислород,Азот,Метан,CO,CO2,Этилен,Этан,Ацетилен,Дибензилсульфид,Напряжение,Поверхностное_Н,Диэлектрическая_жесткость,Содержание_Воды,Срок_службы]])
    prediction = model.predict(data)
    x = int(prediction[0])
    if x >= 56:
        transformer_index = f'Индекс состония трансформатора 1 с процентом оставшегося ресурса {x}'
    elif 29<=x<=55:
        transformer_index = f'Индекс состония трансформатора 2 с процентом оставшегося ресурса {x}'
    elif 17<=x<=28:
        transformer_index = f'Индекс состония трансформатора 3 с процентом оставшегося ресурса {x}'
    elif 6<=x<=16:
        transformer_index = f'Индекс состония трансформатора 4 с процентом оставшегося ресурса {x}'
    elif x<=5:
        transformer_index = f'Индекс состония трансформатора 5 с процентом оставшегося ресурса {x}'

    return transformer_index


def cock():
    myArray = np.array([])
    for n in range(1):
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


def animate(i, xs, ys):

    # Состояние трансформатора с текущего ридинга
    temp_c = cock()

    # Добавляем значения на лист
    xs.append(dt.datetime.now().strftime('%H:%M:%S.%f'))
    ys.append(temp_c)

    # Limit x and y lists to 20 items
    xs = xs[-20:]
    ys = ys[-20:]

    # Draw x and y lists
    ax.clear()
    ax.plot(xs, ys)
    

    # Format plot
    plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title('Показатель Индекса трансформатора')
    plt.ylabel('Индекс состояния трансформатора')
    fig.canvas.draw()


def pack_figure(graph, figure):
    canvas = FigureCanvasTkAgg(figure, graph.Widget)
    plot_widget = canvas.get_tk_widget()
    plot_widget.pack(side='top', fill='both', expand=1)
    return plot_widget

use_agg('TkAgg')

layout = [[sg.Graph((640, 480), (0, 0), (640, 480), key='Graph1'), sg.Graph((640, 480), (0, 0), (640, 480), key='Graph2')]]
window = sg.Window('Matplotlib', layout, finalize=True)


graph1 = window['Graph1']
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
xs = []
ys = []
pack_figure(graph1, fig)
animate(100,xs,ys)

while True:

    event, values = window.read(timeout=10)

    if event == sg.WINDOW_CLOSED:
        break
    elif event == sg.TIMEOUT_EVENT:
        animate(100,xs,ys)
        

window.close()
#ani = animation.FuncAnimation(fig, animate, fargs=(xs, ys), interval=100)
#plt.show()



