import numpy
import pandas
import random
import pprint
import matplotlib.pyplot as plt


#Start class
class Perceptron:
    def __init__(self, eta=0.1, epoch=300):
        self.eta = eta          # learning rate set to 0.1
        self.epoch = epoch      # max number of epochs to iterate to is set to 300
        self.errorList = {}     # epoch number: [errors in epoch, array of weights at that epoch]
        self.w = []


    def dotProductOutput(self, training_vector): # Return o value (1 or -1) after dot prodcut
        o = self.w[0] + numpy.dot(training_vector, self.w[1:])  # add w0 to the equation instead of using in dot prod. w0 is represented as a constant 1
        return numpy.where( o >= 0.0, 1, -1 )             # returns 1s for every o >= 0, else -1 


    def ptr(self, training_vector, target_values, weight_type):
        #training_vector - array of training vectors of the number of examples and the number of flower measurements.
        #target_values - array of target values.
        #weight_type - choosing initial weights

        # 5 weights, because constant x0 = 1, and it also gets a weight w0.
        if weight_type == 0:
            self.w = [0, 0, 0, 0, 0]                           # w - initialize weights to all 0s

        elif weight_type == 1:
            self.w = [1, 1, 1, 1, 1]                           # w - initialize weights to all 1s

        elif weight_type == 2:
            self.w = [0.1, 0.2, 0.3, 0.4, 0.5]                   # w - initialize weights to 4 different numbers between 0 to 1

        elif weight_type == 3:
            self.w = [0, 0, 0, 0, 0]
            for i in self.w:
                self.w[i] = random.random()   # w - initialize weights to 4 random numbers between 0 to 1 


        for i in range(self.epoch):
            errors = 0
            for xi, target in zip(training_vector, target_values):          # for each tuple in (training vector, target values) tuples
                deltaW = self.eta * (target - self.dotProductOutput(xi))    # represent delta delta_w = eta * (t-o)
                self.w[1:] += (deltaW * xi)                                 # delta_wi = delta_w * xi. Thus leading to: wi = wi + delta_wi
                self.w[0:] += deltaW                                        # updating weight for constant x0.

                if deltaW != 0.0:                                           # if delta w has no change, then right match found
                    errors += 1
                
            self.errorList[i] = [errors, self.w] # epoch num: [errors in epoch, end weight of epoch]  <-format for errorList dictionary

            if self.errorList[i][0] == 0: # found epoch with no errors, so last value in dictionary will be epoch if no errors
                break
            
            # Brian FIX THIS CONDITION
            if i > 50:    # if error count wont hit 0, check if the last 3 values are the same
                ele = self.errorList[i][0]
                count = 0
                for x in range(3):
                    if self.errorList[i-x][0] == ele:
                        count+= 1
                if count == 3:
                    break
        return self
#End class




#--------- MAIN ---------

df = pandas.read_csv('iris.data', header=None)
name = df.iloc[0:, 4].values            # getting dataframe of just names
df_measures = df.iloc[:, 0:4].values    # represent the measurements

#----------- TASK 2:
#initial weights set to 0
# LP 1: iris setosa (+1) versus not iris setosa (-1).
p1 = Perceptron()
p2 = Perceptron()
p3 = Perceptron()

<<<<<<< HEAD:test2.py


### Task 2:
# LP 1: iris setosa (+1) versus not iris setosa (-1). Setosa is from start-50
lp1_target = numpy.where(name == 'Iris-setosa', 1, -1) 
lp1 = p1.ptr(df_measures, lp1_target, 0)

x_points_1 = []
y_points_1 = []

print ("*** LP 1 ***")
for i in lp1.errorList.keys():
    print("Epoch " + str(i) + ": " + str(lp1.errorList[i][0]) + " error(s).    Weight: " + str(lp1.errorList[i][1]))
    x_points_1.append(i)
    y_points_1.append(lp1.errorList[i][0])

#print(x_points_1)
#print(y_points_1)
plt.plot(x_points_1, y_points_1, 'ro')
plt.gca().set(title='Number of Errors Per Epoch', xlabel = "Number of Epochs", ylabel='Errors')
plt.savefig('plots1.png')
plt.clf()
# LP 2: iris versicolor (+1) versus not iris versicolor (-1). Versicolor is from 50-100
lp2_target = numpy.where(name == 'Iris-versicolor', 1, -1) 
lp2 = p2.ptr(df_measures, lp2_target, 0)

x_points_2 = []
y_points_2 = []

print ("\n*** LP 2 ***")
for i in lp2.errorList.keys():
    print("Epoch " + str(i) + ": " + str(lp2.errorList[i][0]) + " error(s).    Weight: " + str(lp2.errorList[i][1]) )
    x_points_2.append(i)
    y_points_2.append(lp2.errorList[i][0])
=======
lp1_target = numpy.where(name == 'Iris-setosa', 1, -1) 
lp1 = p1.ptr(df_measures, lp1_target, 0)

print ("*** Task 2: LP 1 ***")
f = open("t2_lp1.txt", "w")
for i in lp1.errorList.keys():
    line = "Epoch " + str(i) + ":    " + str(lp1.errorList[i][0]) + " error(s)    x_0 weight: " + str(lp1.errorList[i][1][0]) + "    iris weights: " + str(lp1.errorList[i][1][1:])
    f.write(line + "\n")
f.close()
print ("Task 2 LP 1 epoch data saved in t2_lp1.txt\n")

'''
elist = lp1.errorList.keys()
print (elist)
#print(elist)
x, y = zip(*elist)
#print (y[0])

plt.plot(x, y.values)
plt.xlabel('Epochs')
plt.ylabel('Number of errors')
plt.show()

# plt.savefig('./perceptron_1.png', dpi=300)
'''

# LP 2: iris versicolor (+1) versus not iris versicolor (-1).
lp2_target = numpy.where(name == 'Iris-versicolor', 1, -1) 
lp2 = p2.ptr(df_measures, lp2_target, 0)

print ("*** Task 2: LP 2 ***")
f = open("t2_lp2.txt", "w")
for i in lp2.errorList.keys():
    line = "Epoch " + str(i) + ":    " + str(lp2.errorList[i][0]) + " error(s)    x_0 weight: " + str(lp2.errorList[i][1][0]) + "    iris weights: " + str(lp2.errorList[i][1][1:])
    f.write(line + "\n")
f.close()
print ("Task 2 LP 2 epoch data saved in t2_lp2.txt\n")
>>>>>>> a0a22522ff3d91c8a76be6bb746bc39efb6574e7:test.py

#print(x_points_2)
#print(y_points_2)
plt.plot(x_points_2, y_points_2, 'b-')
plt.gca().set(title='Number of Errors Per Epoch', xlabel = "Number of Epochs", ylabel='Errors')
plt.savefig('plots2.png')
plt.clf()
 

# LP 3: iris virginica (+1) versus not iris virginica (-1).
lp3_target = numpy.where(name == 'Iris-virginica', 1, -1)
lp3 = p3.ptr(df_measures, lp3_target, 0)

<<<<<<< HEAD:test2.py
x_points_3 = []
y_points_3 = []

print ("\n*** LP 3 ***")
for i in lp3.errorList.keys():
    print("Epoch " + str(i) + ": " + str(lp3.errorList[i][0]) + " error(s).    Weight: " + str(lp3.errorList[i][1]) )
    x_points_3.append(i)
    y_points_3.append(lp3.errorList[i][0])


#print(x_points_3)
#print(y_points_3)
plt.plot(x_points_3, y_points_3, 'b-')
plt.gca().set(title='Number of Errors Per Epoch', xlabel = "Number of Epochs", ylabel='Errors')
plt.savefig('plots3.png')
plt.clf()
=======
print ("*** Task 2: LP 3 ***")
f = open("t2_lp3.txt", "w")
for i in lp3.errorList.keys():
    line = "Epoch " + str(i) + ":    " + str(lp3.errorList[i][0]) + " error(s)    x_0 weight: " + str(lp3.errorList[i][1][0]) + "    iris weights: " + str(lp3.errorList[i][1][1:])
    f.write(line + "\n")
f.close()
print ("Task 2 LP 3 epoch data saved in t2_lp3.txt\n")


'''
print("--------------------")
#----------- TASK 3:
#----- Task 3.1: all initial weights set to 1

p1 = Perceptron()
p2 = Perceptron()
p3 = Perceptron()
# LP1:
lp1 = p1.ptr(df_measures, lp1_target, 1)

print ("\n*** Task 3.1: LP 1 ***")
f = open("t3_1_lp1.txt", "w")
for i in lp1.errorList.keys():
    line = "Epoch " + str(i) + ":    " + str(lp1.errorList[i][0]) + " error(s)    x_0 weight: " + str(lp1.errorList[i][1][0]) + "    iris weights: " + str(lp1.errorList[i][1][1:])
    f.write(line + "\n")
f.close()
print ("Task 3.1 LP 1 epoch data saved in t3_1_lp1.txt")


# LP2:
lp2 = p2.ptr(df_measures, lp2_target, 1)

print ("\n*** Task 3.1: LP 2 ***")
f = open("t3_1_lp2.txt", "w")
for i in lp2.errorList.keys():
    line = "Epoch " + str(i) + ":    " + str(lp2.errorList[i][0]) + " error(s)    x_0 weight: " + str(lp2.errorList[i][1][0]) + "    iris weights: " + str(lp2.errorList[i][1][1:])
    f.write(line + "\n")
f.close()
print ("Task 3.1 LP 2 epoch data saved in t3_1_lp2.txt")


# LP3:
lp3 = p3.ptr(df_measures, lp3_target, 1)

print ("\n*** Task 3.1: LP 3 ***")
f = open("t3_1_lp3.txt", "w")
for i in lp3.errorList.keys():
    line = "Epoch " + str(i) + ":    " + str(lp3.errorList[i][0]) + " error(s)    x_0 weight: " + str(lp3.errorList[i][1][0]) + "    iris weights: " + str(lp3.errorList[i][1][1:])
    f.write(line + "\n")
f.close()
print ("Task 3.1 LP 3 epoch data saved in t3_1_lp3.txt\n")


print("--------------------")
#----- Task 3.2: initial weights set to four different #s between 0 and 1
p1 = Perceptron()
p2 = Perceptron()
p3 = Perceptron()
# LP1:
lp1 = p1.ptr(df_measures, lp1_target, 2)

print ("\n*** Task 3.2: LP 1 ***")
f = open("t3_2_lp1.txt", "w")
for i in lp1.errorList.keys():
    line = "Epoch " + str(i) + ":    " + str(lp1.errorList[i][0]) + " error(s)    x_0 weight: " + str(lp1.errorList[i][1][0]) + "    iris weights: " + str(lp1.errorList[i][1][1:])
    f.write(line + "\n")
f.close()
print ("Task 3.2 LP 1 epoch data saved in t3_2_lp1.txt")


# LP2:
lp2 = p2.ptr(df_measures, lp2_target, 2)

print ("\n*** Task 3.2: LP 2 ***")
f = open("t3_2_lp2.txt", "w")
for i in lp2.errorList.keys():
    line = "Epoch " + str(i) + ":    " + str(lp2.errorList[i][0]) + " error(s)    x_0 weight: " + str(lp2.errorList[i][1][0]) + "    iris weights: " + str(lp2.errorList[i][1][1:])
    f.write(line + "\n")
f.close()
print ("Task 3.2 LP 2 epoch data saved in t3_2_lp2.txt")


# LP3:
lp3 = p3.ptr(df_measures, lp3_target, 2)

print ("\n*** Task 3.2: LP 3 ***")
f = open("t3_2_lp3.txt", "w")
for i in lp3.errorList.keys():
    line = "Epoch " + str(i) + ":    " + str(lp3.errorList[i][0]) + " error(s)    x_0 weight: " + str(lp3.errorList[i][1][0]) + "    iris weights: " + str(lp3.errorList[i][1][1:])
    f.write(line + "\n")
f.close()
print ("Task 3.2 LP 3 epoch data saved in t3_2_lp3.txt\n")


print("--------------------")
#----- Task 3.3: initial weights set to four different random values
p1 = Perceptron()
p2 = Perceptron()
p3 = Perceptron()
# LP1:
lp1 = p1.ptr(df_measures, lp1_target, 3)

print ("\n*** Task 3.3: LP 1 ***")
f = open("t3_3_lp1.txt", "w")
for i in lp1.errorList.keys():
    line = "Epoch " + str(i) + ":    " + str(lp1.errorList[i][0]) + " error(s)    x_0 weight: " + str(lp1.errorList[i][1][0]) + "    iris weights: " + str(lp1.errorList[i][1][1:])
    f.write(line + "\n")
f.close()
print ("Task 3.3 LP 1 epoch data saved in t3_3_lp1.txt")


# LP2:
lp2 = p2.ptr(df_measures, lp2_target, 3)

print ("\n*** Task 3.3: LP 2 ***")
f = open("t3_3_lp2.txt", "w")
for i in lp2.errorList.keys():
    line = "Epoch " + str(i) + ":    " + str(lp2.errorList[i][0]) + " error(s)    x_0 weight: " + str(lp2.errorList[i][1][0]) + "    iris weights: " + str(lp2.errorList[i][1][1:])
    f.write(line + "\n")
f.close()
print ("Task 3.3 LP 2 epoch data saved in t3_3_lp2.txt")


# LP3:
lp3 = p3.ptr(df_measures, lp3_target, 3)

print ("\n*** Task 3.3: LP 3 ***")
f = open("t3_3_lp3.txt", "w")
for i in lp3.errorList.keys():
    line = "Epoch " + str(i) + ":    " + str(lp3.errorList[i][0]) + " error(s)    x_0 weight: " + str(lp3.errorList[i][1][0]) + "    iris weights: " + str(lp3.errorList[i][1][1:])
    f.write(line + "\n")
f.close()
print ("Task 3.3 LP 3 epoch data saved in t3_3_lp3.txt\n")



print("--------------------")
#----------- TASK 4:
#----- Task 4.1: T2 but with iris.data randomly shuffled
df4_1 = df.sample(frac=1)                  # produce a randomly shuffled sample of df. Sample consists of the whole dataframe due to frac=1 
name = df4_1.iloc[0:, 4].values            # getting dataframe of just names
df_measures = df4_1.iloc[:, 0:4].values    # represent the measurements

#initial weights set to 0
# LP 1: iris setosa (+1) versus not iris setosa (-1).
p1 = Perceptron()
p2 = Perceptron()
p3 = Perceptron()

lp1_target = numpy.where(name == 'Iris-setosa', 1, -1) 
lp1 = p1.ptr(df_measures, lp1_target, 0)

print ("*** Task 4.1: LP 1 ***")
f = open("t4_1_lp1.txt", "w")
for i in lp1.errorList.keys():
    line = "Epoch " + str(i) + ":    " + str(lp1.errorList[i][0]) + " error(s)    x_0 weight: " + str(lp1.errorList[i][1][0]) + "    iris weights: " + str(lp1.errorList[i][1][1:])
    f.write(line + "\n")
f.close()
print ("Task 4.1 LP 1 epoch data saved in t4_1_lp1.txt\n")


# LP 2: iris versicolor (+1) versus not iris versicolor (-1).
lp2_target = numpy.where(name == 'Iris-versicolor', 1, -1) 
lp2 = p2.ptr(df_measures, lp2_target, 0)

print ("*** Task 4.1: LP 2 ***")
f = open("t4_1_lp2.txt", "w")
for i in lp2.errorList.keys():
    line = "Epoch " + str(i) + ":    " + str(lp2.errorList[i][0]) + " error(s)    x_0 weight: " + str(lp2.errorList[i][1][0]) + "    iris weights: " + str(lp2.errorList[i][1][1:])
    f.write(line + "\n")
f.close()
print ("Task 4.1 LP 2 epoch data saved in t4_1_lp2.txt\n")


# LP 3: iris virginica (+1) versus not iris virginica (-1).
lp3_target = numpy.where(name == 'Iris-virginica', 1, -1)
lp3 = p3.ptr(df_measures, lp3_target, 0)

print ("*** Task 4.1: LP 3 ***")
f = open("t4_1_lp3.txt", "w")
for i in lp3.errorList.keys():
    line = "Epoch " + str(i) + ":    " + str(lp3.errorList[i][0]) + " error(s)    x_0 weight: " + str(lp3.errorList[i][1][0]) + "    iris weights: " + str(lp3.errorList[i][1][1:])
    f.write(line + "\n")
f.close()
print ("Task 4.1 LP 3 epoch data saved in t4_1_lp3.txt\n")

print("--------------------")

#----- Task 4.2: T2 but with iris.data different randomly shuffled
df4_2 = df4_1.sample(frac=1)               # produce a randomly shuffled sample of df4_1. Sample consists of the whole dataframe due to frac=1 
name = df4_2.iloc[0:, 4].values            # getting dataframe of just names
df_measures = df4_2.iloc[:, 0:4].values    # represent the measurements

#initial weights set to 0
# LP 1: iris setosa (+1) versus not iris setosa (-1).
p1 = Perceptron()
p2 = Perceptron()
p3 = Perceptron()

lp1_target = numpy.where(name == 'Iris-setosa', 1, -1) 
lp1 = p1.ptr(df_measures, lp1_target, 0)

print ("*** Task 4.2: LP 1 ***")
f = open("t4_2_lp1.txt", "w")
for i in lp1.errorList.keys():
    line = "Epoch " + str(i) + ":    " + str(lp1.errorList[i][0]) + " error(s)    x_0 weight: " + str(lp1.errorList[i][1][0]) + "    iris weights: " + str(lp1.errorList[i][1][1:])
    f.write(line + "\n")
f.close()
print ("Task 4.2 LP 1 epoch data saved in t4_2_lp1.txt\n")


# LP 2: iris versicolor (+1) versus not iris versicolor (-1).
lp2_target = numpy.where(name == 'Iris-versicolor', 1, -1) 
lp2 = p2.ptr(df_measures, lp2_target, 0)

print ("*** Task 4.2: LP 2 ***")
f = open("t4_2_lp2.txt", "w")
for i in lp2.errorList.keys():
    line = "Epoch " + str(i) + ":    " + str(lp2.errorList[i][0]) + " error(s)    x_0 weight: " + str(lp2.errorList[i][1][0]) + "    iris weights: " + str(lp2.errorList[i][1][1:])
    f.write(line + "\n")
f.close()
print ("Task 4.2 LP 2 epoch data saved in t4_2_lp2.txt\n")


# LP 3: iris virginica (+1) versus not iris virginica (-1).
lp3_target = numpy.where(name == 'Iris-virginica', 1, -1)
lp3 = p3.ptr(df_measures, lp3_target, 0)

print ("*** Task 4.2: LP 3 ***")
f = open("t4_2_lp3.txt", "w")
for i in lp3.errorList.keys():
    line = "Epoch " + str(i) + ":    " + str(lp3.errorList[i][0]) + " error(s)    x_0 weight: " + str(lp3.errorList[i][1][0]) + "    iris weights: " + str(lp3.errorList[i][1][1:])
    f.write(line + "\n")
f.close()
print ("Task 4.2 LP 3 epoch data saved in t4_2_lp3.txt\n")

'''
>>>>>>> a0a22522ff3d91c8a76be6bb746bc39efb6574e7:test.py