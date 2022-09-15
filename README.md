# MLP_handmande_applied_to_MNIST
This project has the main objective to develop from scratch an mlp and apply it in MNIST data without built-in abstract function. Was used only the library **numpy** for N-Dimensions operation and **matplotlib** for a nice graphic visualization of the results  

## Parameters
The file **global_parameters.py** contains important values that will change the progress of the algorithm, they can be easilly changed  
- rate: Rate learning of the network
- istrain: If True, the network will train
- istest: If True, the network will test
- step_check: (Only for training) Check the progress each 'step_check' iterations
- shape: Shape of the mlp
- showloss: Display loss_figure in screen after the processing
- savefig: Save the loss_figure

Obs: The **shape** is a 3-Dimensional vector, meaning that mlp has 1 input layer, 1 hidden layer and 1 output layer. The quantity of layers and the dimension of the first layer should not be changed, otherwise it may lead at unknown behavior.

## Dependencies
- Matplotlib
- Numpy


## Loss figure example
![loss-output](https://user-images.githubusercontent.com/38757175/190408095-1895569d-03c9-4d08-b794-e0718cabf34e.png)
