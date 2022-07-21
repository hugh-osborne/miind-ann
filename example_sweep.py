import os

for i in range(500):
    os.system('python example_training.py ' + str(i))
    os.system('python example_training.py ' + str(i+1))