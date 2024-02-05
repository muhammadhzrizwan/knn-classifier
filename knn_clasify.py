def knn_clasify(training_data, test, nf, k, math): #Train the model and classify only one input

  distances = []
  k_labels = []

  for i in range(0,len(training_data)):
    sums = 0.0
    for j in range(0,nf):
        sq_diff = (test[j]-training_data[i,j+1])**2
        sums = sums + float(sq_diff)

    dist = math.sqrt(sums)
    distances.append(dist)

  n=0
  d = distances.copy()
  d.sort()

  while n<k:
    for p in range(0,len(distances)):
        if distances[p] == d[n]:
            k_labels.append(training_data[p,0])
    n=n+1

  return max(k_labels,key=k_labels.count)