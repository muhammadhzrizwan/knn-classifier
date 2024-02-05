from knn_clasify import *
def eff(training_data, testing_data, nf, k, np, math): #test the model with the help of testing data and returns the efficiency of model
  test = []
  result = []

  for a in range(0,len(testing_data)):
    test = testing_data[a , 1:]
    result.append(knn_clasify(training_data , test, nf, k, math))

  correct = result==testing_data[: , 0]
  return float(np.count_nonzero(correct)*100/len(testing_data))
