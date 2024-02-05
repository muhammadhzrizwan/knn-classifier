def read(file_name, np): #read the file and return data in desired format i.e. numpy float array
  data_str = np.loadtxt(file_name,delimiter=",", dtype=str)
  data = data_str[1:,:].astype(float)
  return data