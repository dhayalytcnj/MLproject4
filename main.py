'''
Yash D. & Brian H.
CSC 426-01
Project 4 - Perceptrons
'''

import numpy
import pandas

def perceptron(data, n):
    weights = [0,0,0,0]
    inputs = []
    error = []
    eta = 0.1
    iterations = n
    target_flower = ''


#wi <- wi + \Delta wi
'''
Perceptron Training Rule 
1. start with random weight values
2. iteratively apply the perceptron to each training example, changing the perceptron weights whenever the perceptron misclassifies an example
    the weights are changed according to the perceptron training rule: wi <- wi + \Delta wi
    where: \delta wi = \eta * (t - o) * xi, t is the target output for the current training example, o is the perceptron output for the current training example, and \eta is the learning rate (a small positive constant)

    The learning rate is intended to control the degree to which weights are changed at each step. It’s usually set to some small value (e.g., 0.1) and is sometimes made to decay as the # of weight-tuning iterations increases 
3. repeat step 2 as many times as needed until the perceptron classifies all training examples correctly 
'''

if __name__ == "__main__":
    data = pandas.read_csv('iris.data', header = None)

    dataset = []
    features = data[:, :-1]
    print(features)
    for example in data:
        print(example)
        #dataset.append(example[:-1].split(','))
    flower_find = ''
    #for i in dataset:
    #print(data[0])