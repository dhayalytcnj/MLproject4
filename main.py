'''
Yash D. & Brian H.
CSC 426-01
Project 4 - Perceptrons
'''


'''
Perceptron Training Rule 
1. start with random weight values
2. iteratively apply the perceptron to each training example, changing the perceptron weights whenever the perceptron misclassifies an example
    the weights are changed according to the perceptron training rule: wi <- wi + \Delta wi
    where: \delta wi = \eta * (t - o) * xi, t is the target output for the current training example, o is the perceptron output for the current training example, and \eta is the learning rate (a small positive constant)

    The learning rate is intended to control the degree to which weights are changed at each step. It’s usually set to some small value (e.g., 0.1) and is sometimes made to decay as the # of weight-tuning iterations increases 
3. repeat step 2 as many times as needed until the perceptron classifies all training examples correctly 
'''



import numpy
import pandas

class Perceptron:
    def __init__(self, eta = 0.1):
        self.weights = None
        self.delta_w = 0.0
        self.iterations = iterations
        self.iteration_stop = []
        self.eta = eta
        self.misclassified = []
    

    def output(self, x: numpy.array):   # output function that reads x which is preset as an array
        o = float( numpy.dot(x, self.weights) + self.delta_w )
        return o
    

    def prediction(self, x: numpy.array):
        #ask Dr Bloodgood should 0 be considered negative or positive 
        result = 0
        if self.output(x) >= 0.0:
            result = 1
        else:
            result = -1
        return result

    def Perceptron_Training(self, x: nump.array, y:numpy.array, iterations = 200):
        # o is training samples for the model
        # t is labeling the training samples
        self.delta_w = 0.0
        self.weights = numpy.zeros(x.shape[1])
        self.misclassified = []

        for i in range(iterations):
            error_count = 0
            iteration_count = 1
            for x_i, y_i in zip(x, y):
                change = self.eta * (y_i - self.prediction(x_i) ) * x_i
                self.delta_w += change
                self.weights += change
                if change != 0.0:
                    error_count += 1
                    interation_count += 1
                else
                    break
            self.misclassified.append(error_count)
            self.iteration_stop.append(iteration_count)
            if error_count == 0:



if __name__ == "__main__":
    data = pandas.read_csv('iris.data', header = None)
    
    print

     for example in data:
        dataset.append(example)
    
