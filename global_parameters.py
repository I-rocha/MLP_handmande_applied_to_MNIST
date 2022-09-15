# rate: rate learning
# istrain: if traing is supposed to happened
# istest: if test will happened
# step_check (Only for training): step size to show progress
# shape: Shape of the MLP

parameters = {'rate': 0.1,
              'istrain': False,
              'istest': True,
              'step_check': 10000,
              'shape': (28*28, 100, 10),
              'showloss' : True,
              'savefig' : True
              }
