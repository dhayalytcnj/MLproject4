import numpy
import pandas
import random
import pprint
import matplotlib.pyplot as plt


#Start class
class Perceptron:
    def __init__(self, eta=0.01, epoch=200):
        #epoch - int number of passes over the training dataset.
        self.eta = eta
        self.epoch = epoch
        self.errorList = {} # epoch number: [errors in epoch, array of weights at that epoch]


    def dotProductOutput(self, training_vector): # Return o value (1 or -1) after dot prodcut
        o = 1 + numpy.dot(training_vector, self.w[0:])  # add w0 to the equation instead of using in dot prod. w0 is represented as a constant 1
        return numpy.where(o >= 0.0, 1, -1)             # returns 1s for every o >= 0, else -1 


    def ptr(self, training_vector, target_values, weight_type):
        #training_vector - array of training vectors of the number of examples and the number of flower measurements.
        #target_values - array of target values.

        if weight_type == 0:
            self.w = [0, 0, 0, 0]                           # w - initialize weights to all 0s

        elif weight_type == 1:
            self.w = [1, 1, 1, 1]                           # w - initialize weights to all 1s

        elif weight_type == 2:
            self.w = [0.1, 0.2, 0.3, 0.4]                   # w - initialize weights to 4 different numbers between 0 to 1

        elif weight_type == 3:
            self.w = [0, 0, 0, 0]
            for i in self.w:
                self.w[i] = round(np.random.random(), 2)    # w - initialize weights to 4 random numbers between 0 to 1


        for i in range(self.epoch):
            errors = 0
            for xi, target in zip(training_vector, target_values):          # for each tuple in (training vector, target values) tuples
                deltaW = self.eta * (target - self.dotProductOutput(xi))    # represent delta delta_w = eta * (t-o)
                self.w[0:] += (deltaW * xi)                                 # delta_wi = delta_w * xi. Thus leading to: wi = wi + delta_wi

                if deltaW != 0.0:                                           # if delta w has no change, then right match found
                    errors += 1
                
            self.errorList[i] = [errors, self.w] # epoch num: [errors in epoch, end weight of epoch]  <-format for errorList dictionary

            if self.errorList[i][0] == 0: # found epoch with no errors, so last value in dictionary will be epoch if no errors
                break
            
            # Brian FIX THIS CONDITION
            
            if i > 50:    # if error count wont hit 0, check if the last 5 values are the same
                ele = self.errorList[i][0]
                temp_count = 0

                for x in range(5):
                    if self.errorList[i-x][0] == ele:
                        temp_count += 1
                if temp_count == 5:
                    break
        return self
#End class


#main
df = pandas.read_csv('iris.data', header=None)
name = df.iloc[0:, 4].values #getting dataframe of just names
df_measures = df.iloc[:, 0:4].values
#print (df_measures)

p = Perceptron()

### Task 2:
# LP 1: iris setosa (+1) versus not iris setosa (-1). Setosa is from start-50
lp1_target = numpy.where(name == 'Iris-setosa', 1, -1) 
lp1 = p.ptr(df_measures, lp1_target, 0)

print ("*** LP 1 ***")
for i in lp1.errorList.keys():
    print("Epoch " + str(i) + ": " + str(p.errorList[i][0]) + " error(s).    Weight: " + str(p.errorList[i][1]) )


# LP 2: iris versicolor (+1) versus not iris versicolor (-1). Versicolor is from 50-100
lp2_target = numpy.where(name == 'Iris-versicolor', 1, -1) 
lp2 = p.ptr(df_measures, lp2_target, 0)

print ("\n*** LP 2 ***")
for i in lp2.errorList.keys():
    print("Epoch " + str(i) + ": " + str(p.errorList[i][0]) + " error(s).    Weight: " + str(p.errorList[i][1]) )


# LP 3: iris virginica (+1) versus not iris virginica (-1). Virginica is from 100-end
lp3_target = numpy.where(name == 'Iris-virginica', 1, -1)
lp3 = p.ptr(df_measures, lp3_target, 0)

print ("\n*** LP 3 ***")
for i in lp3.errorList.keys():
    print("Epoch " + str(i) + ": " + str(p.errorList[i][0]) + " error(s).    Weight: " + str(p.errorList[i][1]) )
#print (lp3.errorList[99 - 5][0])