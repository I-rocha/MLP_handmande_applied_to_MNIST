import numpy as np
from global_parameters import parameters
import matplotlib.pyplot as plt

def loss_plot():
    path = 'output/Loss-output.txt'
    fd = open(path)
    y_axis = []
    x_axis = []
    cont = 0
    xi = 0
    for line in fd:
        if cont % 50 == 0:
            y_axis.append(float(line))
            x_axis.append(xi)
            xi += 1
            
        cont += 1
        
    fd.close()
    plt.plot(x_axis, y_axis)
    
    if parameters['savefig']:
        plt.savefig('output/loss-output.png')

    plt.show()

    
