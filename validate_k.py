from knn_clasify import *
def validate_k(training_data, validation_data, nf, np, math): #uses validation data to set optimum value of K
  effs=[]
  for i in range(1,20,2):
    test = []
    result = []
    check = (2*i)+1
    for a in range(0,len(validation_data)):
      test = validation_data[a , 1:]
      result.append(knn_clasify(training_data , test, nf, check, math))
    correct = result==validation_data[: , 0]
    eff = float(np.count_nonzero(correct)*100/len(validation_data))
    effs.append(eff)
  ind = effs.index(max(effs))
  return (ind*2)+1