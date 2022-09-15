from global_parameters import parameters as p
from graphic import loss_plot
import time

# Main
from data import preprocess
from mlp import MLP


if __name__ == '__main__':
    istrain = p['istrain']
    istest = p['istest']
    showloss = ['showloss']
    
    dx = preprocess()

    print('Training: {0}\n'
          'Test: {1}'
          .format(str(istrain), str(istest))
          )
    
    mlp = MLP(rate = p['rate'], shape = p['shape'])
    
    
    if istrain:
        print('#' *50)
        print('Starting train....')
        tds = time.process_time()
        mlp.training(dx.train, dx.train_labels)
        tdf = time.process_time()
        
        print('\nTraining_time: {0}'.format(tdf-tds))
    
    if istest:
        print('#' *50)
        print('Starting test....')
        tds = time.process_time()
        acc, err = mlp.test(dx.test, dx.test_labels)
        tdf = time.process_time()
        
        print('\nTest_time: {0}'.format(tdf-tds))
        print('Test rate: {0}'.format(acc))

    if showloss:
        loss_plot()
    
    print('#'*50 + '\nEOF')
