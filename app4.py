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
from statsmodels.tsa.ar_model import AutoReg
import textwrap
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
    a = random.randint(1,10000)
    b = random.randint(1,10000)
    c = random.randint(1,10000)
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
        
    myArray = transformer(a,b,c,d,d,e,f,g,h,j,k,l,z,m,v)
    #print(myArray)
        
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
    ax2.plot(xs1, ys1, 'k')
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

def LEDIndicator1(key=None, radius=50):
    return sg.Graph(canvas_size=(radius, radius),
             graph_bottom_left=(-radius, -radius),
             graph_top_right=(radius, radius),
             pad=(0, 0), key=key)



def SetLED(window, key, color):
    graph = window[key]
    graph.erase()
    graph.draw_circle((0, 0), 48, fill_color=color, line_color=color)

def SetLED1(window, key, color):
    graph = window[key]
    graph.erase()
    graph.draw_circle((0, 0), 30, fill_color=color, line_color=color)

color = '#68748c'
def color1():
    global color
    global myArray
    if myArray ==5:
        color = '#f50707'
    else: color ='#68748c'
    return color




layout = [[sg.Graph((640, 480), (0, 0), (640, 480), key='Graph1'),sg.Graph((640, 480), (0, 0), (640, 480), key='Graph2'), sg.Graph((640, 480), (0, 0), (640, 480), key='Graph3')], [sg.Text('Индикатор состояния трансформатора', size=(30,1)), [sg.Text('Индекс состояния'), LEDIndicator('_cpu1_')], [LEDIndicator1('_cpu_'), sg.Text('Отлично')], [LEDIndicator1('_ram_'), sg.Text('Хорошо')], [LEDIndicator1('_temp_'), sg.Text('Внимание')], [LEDIndicator1('_server1_'), sg.Text('Неисправность')],[sg.Text(font=('Helvetica', 15), key='-TEXT1-', text_color='black'),sg.Text(font=('Helvetica', 12), key='-TEXT2-', text_color='black'),sg.Text(font=('Helvetica', 12), key='-TEXT3-', text_color='black'),sg.Text(font=('Helvetica', 12), key='-TEXT4-', text_color='black'),sg.Text(font=('Helvetica', 12), key='-TEXT5-', text_color='black'),sg.Text(font=('Helvetica', 12), key='-TEXT6-', text_color='black'),sg.Text(font=('Helvetica', 12), key='-TEXT7-', text_color='black')]]],[sg.Button('Пауза'), sg.Button('Выход')]
window = sg.Window('Индекс состояния трансформатора', layout, finalize=True, background_color=color)


#
def sound1():
    global myArray
    if myArray ==5:
        duration = 1000  # milliseconds
        freq = 440  # Hz
        winsound.Beep(freq, duration)


yhat_pred = 0
yhat=[]
count1 = []




#def count1():
    #global count1
    #count1.append(1)
    #return count

def pred():
    global ys
    global yhat
    global yhat_pred
    if np.count_nonzero(ys) > 10:
        data = ys
        #print(data)
        model = AutoReg(data, lags=1)
        model_fit = model.fit()
        yhat = model_fit.predict(len(data), len(data))
        yhat_pred=yhat[0]
        print(yhat_pred)
        #print(yhat)
    else:
        yhat_pred = 0
    print(yhat_pred)
    return (yhat_pred)


def animate2(i,xs2, ys2):

    # Состояние трансформатора с текущего ридинга
    temp_c = yhat_pred
    

    # Добавляем значения на лист
    xs2.append(dt.datetime.now().strftime('%H:%M:%S.%f'))
    ys2.append(temp_c)
    #print(ys2)

    # Limit x and y lists to 20 items
    xs2 = xs2[-20:]
    ys2 = ys2[-20:]

    # Draw x and y lists
    ax3.clear()
    ax3.plot(xs2, ys2, 'm')
    ax3.set_title("Предсказание индекса трансформатора")
    ax3.set_xlabel("Время")
    ax3.set_ylabel("Индекс")
    ax3.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off

    

    # Format plot
    
    fig3.canvas.draw()

def pack_figure(graph, figure):
    canvas = FigureCanvasTkAgg(figure, graph.Widget)
    plot_widget = canvas.get_tk_widget()
    plot_widget.pack(side='top', fill='both', expand=1)
    return plot_widget



xs = []
ys = []
xs1= []
ys1 = []
xs2 = []
ys2 = []


graph1 = window['Graph1']
graph2 = window['Graph2']
graph3 = window['Graph3']
                      # Turn the interactive mode off
fig1 = plt.figure(1)                # Create a new figure
ax1 = plt.subplot(111)              # Add a subplot to the current figure.
fig2 = plt.figure(2)                # Create a new figure
ax2 = plt.subplot(111)
fig3 = plt.figure(3)                # Create a new figure
ax3 = plt.subplot(111)                 # Add a subplot to the current figure.
pack_figure(graph1, fig1)           # Pack figure under graph
pack_figure(graph2, fig2)
pack_figure(graph3, fig3)
animate(100,xs,ys)
animate1(100,xs1,ys1)
animate2(100,xs2, ys2)


def cvet(): 
    color=None
    if myArray == 1:
        color = 'green'
    elif myArray == 2:
        color ='yellow'
    elif myArray == 3:
        color = 'orange'
    elif myArray ==4:
        color = 'red'
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

def pop():
    global myArray
    if myArray == 5:
        title = "Индекс состояния 5"
        message = "Необходимо обратать внимание на состояние устройства"
        display_notification(title, message, img_error, 10000, use_fade_in=True)


SE_FADE_IN = True
WIN_MARGIN = 60

# colors
WIN_COLOR = "#282828"  #282828
TEXT_COLOR = "#ffffff"

DEFAULT_DISPLAY_DURATION_IN_MILLISECONDS = 10000

# Base64 Images to use as icons in the window
img_error = b'iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAMAAABEpIrGAAAAA3NCSVQICAjb4U/gAAAACXBIWXMAAADlAAAA5QGP5Zs8AAAAGXRFWHRTb2Z0d2FyZQB3d3cuaW5rc2NhcGUub3Jnm+48GgAAAIpQTFRF////20lt30Bg30pg4FJc409g4FBe4E9f4U9f4U9g4U9f4E9g31Bf4E9f4E9f4E9f4E9f4E9f4FFh4Vdm4lhn42Bv5GNx5W575nJ/6HqH6HyI6YCM6YGM6YGN6oaR8Kev9MPI9cbM9snO9s3R+Nfb+dzg+d/i++vt/O7v/fb3/vj5//z8//7+////KofnuQAAABF0Uk5TAAcIGBktSYSXmMHI2uPy8/XVqDFbAAAA8UlEQVQ4y4VT15LCMBBTQkgPYem9d9D//x4P2I7vILN68kj2WtsAhyDO8rKuyzyLA3wjSnvi0Eujf3KY9OUP+kno651CvlB0Gr1byQ9UXff+py5SmRhhIS0oPj4SaUUCAJHxP9+tLb/ezU0uEYDUsCc+l5/T8smTIVMgsPXZkvepiMj0Tm5txQLENu7gSF7HIuMreRxYNkbmHI0u5Hk4PJOXkSMz5I3nyY08HMjbpOFylF5WswdJPmYeVaL28968yNfGZ2r9gvqFalJNUy2UWmq1Wa7di/3Kxl3tF1671YHRR04dWn3s9cXRV09f3vb1fwPD7z9j1WgeRgAAAABJRU5ErkJggg=='
img_success = b'iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAMAAABEpIrGAAAAA3NCSVQICAjb4U/gAAAACXBIWXMAAAEKAAABCgEWpLzLAAAAGXRFWHRTb2Z0d2FyZQB3d3cuaW5rc2NhcGUub3Jnm+48GgAAAHJQTFRF////ZsxmbbZJYL9gZrtVar9VZsJcbMRYaMZVasFYaL9XbMFbasRZaMFZacRXa8NYasFaasJaasFZasJaasNZasNYasJYasJZasJZasJZasJZasJZasJYasJZasJZasJZasJZasJaasJZasJZasJZasJZ2IAizQAAACV0Uk5TAAUHCA8YGRobHSwtPEJJUVtghJeYrbDByNjZ2tvj6vLz9fb3/CyrN0oAAADnSURBVDjLjZPbWoUgFIQnbNPBIgNKiwwo5v1fsQvMvUXI5oqPf4DFOgCrhLKjC8GNVgnsJY3nKm9kgTsduVHU3SU/TdxpOp15P7OiuV/PVzk5L3d0ExuachyaTWkAkLFtiBKAqZHPh/yuAYSv8R7XE0l6AVXnwBNJUsE2+GMOzWL8k3OEW7a/q5wOIS9e7t5qnGExvF5Bvlc4w/LEM4Abt+d0S5BpAHD7seMcf7+ZHfclp10TlYZc2y2nOqc6OwruxUWx0rDjNJtyp6HkUW4bJn0VWdf/a7nDpj1u++PBOR694+Ftj/8PKNdnDLn/V8YAAAAASUVORK5CYII='

# -------------------------------------------------------------------

def display_notification(title, message, icon, display_duration_in_ms=DEFAULT_DISPLAY_DURATION_IN_MILLISECONDS, use_fade_in=True, alpha=0.9, location=None):
    """
    Function that will create, fade in and out, a small window that displays a message with an icon
    The graphic design is similar to other system/program notification windows seen in Windows / Linux
    :param title: (str) Title displayed at top of notification
    :param message: (str) Main body of the noficiation
    :param icon: (str) Base64 icon to use. 2 are supplied by default
    :param display_duration_in_ms: (int) duration for the window to be shown
    :param use_fade_in: (bool) if True, the window will fade in and fade out
    :param alpha: (float) Amount of Alpha Channel to use.  0 = invisible, 1 = fully visible
    :param location: Tuple[int, int] location of the upper left corner of window. Default is lower right corner of screen
    """

    # Compute location and size of the window
    message = textwrap.fill(message, 50)
    win_msg_lines = message.count("\n") + 1

    screen_res_x, screen_res_y = sg.Window.get_screen_size()
    win_margin = WIN_MARGIN  # distance from screen edges
    win_width, win_height = 864, 566 + (14.8 * win_msg_lines)
    win_location = location if location is not None else (screen_res_x - screen_res_x/2, screen_res_y - screen_res_y/2)

    layout = [[sg.Graph(canvas_size=(win_width, win_height), graph_bottom_left=(0, win_height), graph_top_right=(win_width, 0), key="-GRAPH-",
                        background_color=WIN_COLOR, enable_events=True)]]

    window = sg.Window(title, layout, background_color=WIN_COLOR, no_titlebar=True,
                       location=win_location, keep_on_top=True, alpha_channel=0, margins=(0, 0), element_padding=(0, 0),
                       finalize=True)

    window["-GRAPH-"].draw_rectangle((win_width, win_height), (-win_width, -win_height), fill_color=WIN_COLOR, line_color=WIN_COLOR)
    window["-GRAPH-"].draw_image(data=icon, location=(20, 20))
    window["-GRAPH-"].draw_text(title, location=(64, 20), color=TEXT_COLOR, font=("Arial", 24, "bold"), text_location=sg.TEXT_LOCATION_TOP_LEFT)
    window["-GRAPH-"].draw_text(message, location=(64, 44), color=TEXT_COLOR, font=("Arial", 26), text_location=sg.TEXT_LOCATION_TOP_LEFT)

    # change the cursor into a "hand" when hovering over the window to give user hint that clicking does something
    window['-GRAPH-'].set_cursor('hand2')

    if use_fade_in == True:
        for i in range(1,int(alpha*100)):               # fade in
            window.set_alpha(i/100)
            event, values = window.read(timeout=20)
            if event != sg.TIMEOUT_KEY:
                window.set_alpha(1)
                break
        event, values = window(timeout=display_duration_in_ms)
        if event == sg.TIMEOUT_KEY:
            for i in range(int(alpha*100),1,-1):       # fade out
                window.set_alpha(i/100)
                event, values = window.read(timeout=20)
                if event != sg.TIMEOUT_KEY:
                    break
    else:
        window.set_alpha(alpha)
        event, values = window(timeout=display_duration_in_ms)

    window.close()








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
    pred()
    pop() 
    sound1()
    
    event, values = window.read(timeout=10)
    print(event, values)
    

    

    if event in (None, 'Выход'):
            break
    elif event == sg.TIMEOUT_EVENT:
        pred()
        
        procg()
        #sohr()
        #logs()
        cock()
        animate(100,xs,ys)
        animate1(100,xs1,ys1)
        animate2(100,xs2, ys2)
        
    SetLED(window, '_cpu1_', cvet())
    SetLED1(window, '_cpu_', 'green')
    SetLED1(window, '_ram_', 'yellow')
    SetLED1(window, '_temp_', 'orange')
    SetLED1(window, '_server1_', 'red')
    window['-TEXT1-'].update(f"Индекс состояния {myArray}")
    window['-TEXT2-'].update(f"Водород {a} ppm")
    window['-TEXT3-'].update(f"Кислород {b} ppm")
    window['-TEXT4-'].update(f"Азот {c} ppm")
    window['-TEXT5-'].update(f"Метан {d} ppm")
    window['-TEXT6-'].update(f"CO {e} ppm")
    window['-TEXT7-'].update(f"CO2 {f} ppm")
    
   
     

window.close()