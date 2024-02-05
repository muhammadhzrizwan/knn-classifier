import numpy as np 
import math
from read import *
from knn_clasify import *
from validate_k import *
from eff import *

add = input("Please provide adress to the data set: ")
data = read(add, np)

training_data = data[0:int(len(data)*0.6) , :]
validation_data = data[len(training_data):int(len(training_data)+len(data)*0.2) , :]
testing_data = data[int(-len(data)*0.2):-1 , :]

nf = data.shape[1]-1

do = input("\n1. Compute the validated value of \'k\' and efficiency against that\n2. Set your own value of \'k\' and compute efficiency\n3. Get a predicted value against your input\n4. Exit\nSelect an option from the menu: ")

#class_file_name_1 = "/content/drive/MyDrive/Data Sets/class1.csv"
#class_file_name_2 = "/content/drive/MyDrive/Data Sets/class2.csv"

if do=='1':
  k = validate_k(training_data, validation_data, nf, np, math)
  efficiency = eff(training_data, testing_data, nf, k, np, math)
  print(f"\nValidated value of \'k\' is {k} and efficiency against validated model is {efficiency}")

elif do=='2':
  k = int(input("\nEnter value of \'k\' : "))
  efficiency = eff(training_data, testing_data, nf, k, np, math)
  print(f"Efficiency of the model against given value of \'k\' is {efficiency}")

elif do=='3':
  test = []
  print()
  for j in range(0, nf):
   t = float(input(f"Enter value of feature # {j+1}: "))
   test.append(t)

  do_k = input("\n1. Use validated value of \'k\'\n2. Input your own \'k\'\nPlease choose from above: ")

  if do_k=='1':
    k = validate_k(training_data, validation_data, nf, np, math)
  elif do_k=='2':
    k = int(input("\nEnter value of \'k\' : "))
  print(f"\nPredicted result is \"{int(knn_clasify(training_data, test, nf, k, math))}\" against k = {k}'")