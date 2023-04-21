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
import winsound

model = load_model('model.h5')

def transformer(Водород,Кислород,Азот,Метан,CO,CO2,Этилен,Этан,Ацетилен,Дибензилсульфид,Напряжение,Поверхностное_Н,Диэлектрическая_жесткость,Содержание_Воды,Срок_службы):
    data = np.array([[Водород,Кислород,Азот,Метан,CO,CO2,Этилен,Этан,Ацетилен,Дибензилсульфид,Напряжение,Поверхностное_Н,Диэлектрическая_жесткость,Содержание_Воды,Срок_службы]])
    prediction = model.predict(data)
    x = int(prediction[0])
    if x >= 56:
        transformer_index = 1
    elif 29<=x<=55:
        transformer_index = 2
    elif 17<=x<=28:
        transformer_index = 3
    elif 6<=x<=16:
        transformer_index = 4   
    elif x<=5:
        transformer_index = 5

    return transformer_index

def proc(Водород,Кислород,Азот,Метан,CO,CO2,Этилен,Этан,Ацетилен,Дибензилсульфид,Напряжение,Поверхностное_Н,Диэлектрическая_жесткость,Содержание_Воды,Срок_службы):
    data = np.array([[Водород,Кислород,Азот,Метан,CO,CO2,Этилен,Этан,Ацетилен,Дибензилсульфид,Напряжение,Поверхностное_Н,Диэлектрическая_жесткость,Содержание_Воды,Срок_службы]])
    prediction = model.predict(data)
    x = int(prediction[0])


    return x


proca = 0

def procg():
    global proca
    a = random.randint(7000,10000)
    b = random.randint(7000,10000)
    c = random.randint(7000,10000)
    d = random.randint(7000,10000)
    e = random.randint(7000,10000)
    f = random.randint(7000,10000)
    g = 1000
    h = 1000
    j = 1000
    k = 1000
    l = 70
    z = 40
    m = 50
    v = 20
        
    proca = proc(a,b,c,d,d,e,f,g,h,j,k,l,z,m,v)
    
        
    return(proca)

myArray=0
a = 0
b = 0
c=0
d=0
e=0
f=0

def cock():
    global myArray
    global a
    global b
    global c
    global d
    global e
    global f
    a = random.randint(7000,10000)
    b = random.randint(7000,10000)
    c = random.randint(10,10000)
    d = random.randint(10,10000)
    e = random.randint(10,10000)
    f = random.randint(10,10000)
    g = 1000
    h = 1000
    j = 1000
    k = 1000
    l = 70
    z = 40
    m = 50
    v = 20
        
    myArray = transformer(a,b,c,d,d,e,f,g,h,j,k,l,z,m,v)
        
    return(myArray)





def animate(i, xs, ys):

    # Состояние трансформатора с текущего ридинга
    temp_c = myArray

    # Добавляем значения на лист
    xs.append(dt.datetime.now().strftime('%H:%M:%S.%f'))
    ys.append(temp_c)

    # Limit x and y lists to 20 items
    xs = xs[-20:]
    ys = ys[-20:]

    # Draw x and y lists
    ax1.clear()
    ax1.plot(xs, ys)
    ax1.set_title("Индекс состояния трансформатора")
    ax1.set_xlabel("Время")
    ax1.set_ylabel("Индекс")
    ax1.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off

    

    # Format plot
    
    fig1.canvas.draw()

def animate1(i, xs1, ys1):

    #Состояние трансформатора с текущего ридинга
    temp_c = proca

    # Добавляем значения на лист
    xs1.append(dt.datetime.now().strftime('%H:%M:%S.%f'))
    ys1.append(temp_c)

    #Limit x and y lists to 20 items
    xs1 = xs1[-20:]
    ys1 = ys1[-20:]

    #Draw x and y lists
    ax2.clear()
    ax2.plot(xs1, ys1)
    ax2.set_title("Процент состояния")
    ax2.set_xlabel("Время")
    ax2.set_ylabel("Процент индекса")
    ax2.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off

    

    # Format plot

    fig2.canvas.draw()

def pack_figure(graph, figure):
    canvas = FigureCanvasTkAgg(figure, graph.Widget)
    plot_widget = canvas.get_tk_widget()
    plot_widget.pack(side='top', fill='both', expand=1)
    return plot_widget

use_agg('TkAgg')

def LEDIndicator(key=None, radius=100):
    return sg.Graph(canvas_size=(radius, radius),
             graph_bottom_left=(-radius, -radius),
             graph_top_right=(radius, radius),
             pad=(0, 0), key=key)

def SetLED(window, key, color):
    graph = window[key]
    graph.erase()
    graph.draw_circle((0, 0), 48, fill_color=color, line_color=color)

color = '#68748c'
def color1():
    global color
    global myArray
    if myArray ==5:
        color = '#f50707'
    else: color ='#68748c'
    return color

layout = [[sg.Graph((640, 480), (0, 0), (640, 480), key='Graph1'),sg.Graph((640, 480), (0, 0), (640, 480), key='Graph2')], [sg.Text('Индикатор состояния трансформатора', size=(30,1))],[sg.Text('Индекс состояния'), LEDIndicator('_cpu_'),[sg.Text(font=('Helvetica', 15), key='-TEXT1-', text_color='black')],[sg.Text(font=('Helvetica', 12), key='-TEXT2-', text_color='black')],[sg.Text(font=('Helvetica', 12), key='-TEXT3-', text_color='black')],[sg.Text(font=('Helvetica', 12), key='-TEXT4-', text_color='black')],[sg.Text(font=('Helvetica', 12), key='-TEXT5-', text_color='black')],[sg.Text(font=('Helvetica', 12), key='-TEXT6-', text_color='black')],[sg.Text(font=('Helvetica', 12), key='-TEXT7-', text_color='black')]]],[sg.Button('Пауза'), sg.Button('Выход')]
window = sg.Window('Индекс состояния трансформатора', layout, finalize=True, background_color=color)


#
def sound1():
    global myArray
    if myArray ==5:
        duration = 1000  # milliseconds
        freq = 440  # Hz
        winsound.Beep(freq, duration)





xs = []
ys = []
xs1= []
ys1 = []

graph1 = window['Graph1']
graph2 = window['Graph2']
                      # Turn the interactive mode off
fig1 = plt.figure(1)                # Create a new figure
ax1 = plt.subplot(111)              # Add a subplot to the current figure.
fig2 = plt.figure(2)                # Create a new figure
ax2 = plt.subplot(111)              # Add a subplot to the current figure.
pack_figure(graph1, fig1)           # Pack figure under graph
pack_figure(graph2, fig2)
animate(100,xs,ys)
animate1(100,xs1,ys1)



def cvet(): 
    color=None
    if myArray == 1:
        color = 'green'
    elif myArray == 2:
        color ='blue'
    elif myArray == 3:
        color = 'yellow'
    elif myArray ==4:
        color = 'orange'
    elif myArray==5:
        color = 'red'

    return color

def index():
   
    if myArray == 1:
        transformer_index = f'Индекс состояния трансформатора 1'
    elif myArray == 2:
        transformer_index = f'Индекс состояния трансформатора 2'
    elif myArray == 3:
        transformer_index = f'Индекс состояния трансформатора 3'
    elif myArray ==4:
        transformer_index = f'Индекс состояния трансформатора 4'
    elif myArray==5:
        transformer_index = f'Индекс состояния трансформатора 5'
    return transformer_index

#def logs():
    f = open("log.txt", "a")
    f.write('\n')
    f.write(str(index()))
    f.write('  ')
    f.write(str(dt.datetime.now()))
    f.close()

data = np.array([0])
x=np.array([0])
y=np.array([0])


#def sohr():
    #global myArray
    #global data
    #global xs
    #global ys
    #global y
    #global x
    #y=np.append(y, ys)
    #x=np.append(x, xs)
    #df = pd.DataFrame({"date" : x, "index" : y})
    #df.to_csv("sub1.csv", index=False)
    #data = np.append(data, myArray)
    #print (x,y)
    #return data
    


while True:
    procg()
    cock()  
    event, values = window.read(timeout=10)
    print(event, values)

    if event in (None, 'Выход'):
            break
    elif event == sg.TIMEOUT_EVENT:
        procg()
        #sohr()
        #logs()
        cock()
        animate(100,xs,ys)
        animate1(100,xs1,ys1)
        sound1()
    SetLED(window, '_cpu_', cvet())
    window['-TEXT1-'].update(f"Индекс состояния {myArray}")
    window['-TEXT2-'].update(f"Водород {a} ppm")
    window['-TEXT3-'].update(f"Кислород {b} ppm")
    window['-TEXT4-'].update(f"Азот {c} ppm")
    window['-TEXT5-'].update(f"Метан {d} ppm")
    window['-TEXT6-'].update(f"CO {e} ppm")
    window['-TEXT7-'].update(f"CO2 {f} ppm")
   
     

window.close()