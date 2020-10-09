from BayesLSTM import BNLSTM
from BayesSiren import BNSIREN
import pandas as pd
import os
import numpy as np
import scipy
import scipy.io
from scipy.fft import fft
import matplotlib.pyplot as plt
import random
from tqdm import tqdm

def load_chattering_data():
    split_size = 100
    df = pd.DataFrame(columns=['Time', 'Amplitude', 'Label'])
    data = []
    print(df.head())
    for fld in os.listdir('cutting_tests_processed/'):
        for fil in os.listdir('cutting_tests_processed/' + fld):
            print('cutting_tests_processed/' + fld + '/' + fil)
            out = scipy.io.loadmat('cutting_tests_processed/' + fld + '/' + fil)['tsDS']
            splits = out.shape[0] / split_size
            split_arr = np.array_split(out, splits, axis=0)
            for splar in split_arr[0:-1]:
                data.append([splar[:split_size, 0], splar[:split_size, 1], fil.split('_')[0]])
        print('Completed', len(data))
    return data

def error_func(err, theta):
    def error1(x):
        return theta * np.sin(15000 * x)

    def error2(x):
        return theta * np.sin(13000 * x)

    def error3(x):
        return theta * np.cos(20000 * x)

    def error4(x):
        return theta * np.cos(17500 * x)

    if err == "Error1":
        return error1
    if err == "Error2":
        return error2
    if err == "Error3":
        return error3
    if err == "Error4":
        return error4


data = load_chattering_data()
random.shuffle(data)
for i in range(len(data)):
    if data[i][2] == "c":
        data[i][2] = "Error1"
    if data[i][2] == "i":
        data[i][2] = "Error2"
    if data[i][2] == "u":
        data[i][2] = "Error3"
    if data[i][2] == "s":
        data[i][2] = "Error4"


def normal_func(sensor):
    def normal_S1(x):
        return 1* np.sin(x)
    if sensor == "S1":
        return normal_S1


bn = BNLSTM(sensor_list=["S1"], errors={"S1": ["Error1", "Error2", "Error3", "Error4"]},
                prior_distributions={"Error1": error_func("Error1", 0.1), "Error2": error_func("Error2" , 0.15),
                                     "Error3": error_func("Error3", 0.03), "Error4": error_func("Error4", 0.08)}, sigmas={"S1": 1} ,
             compound=False, fft=True)

ITER = 0
mse_sub = []
acc = 0
count = 0
data =data[0:10000]

acc_list = []
for dt in tqdm(data):
   time = dt[0]
   input = dt[1]
   label = dt[2]
   count +=1

   sensor_inputs = {"S1": input}
   err, probs, mse = bn.bayesian_calculation_update(sensor_inputs, time, ITER)
   #print(err)
   #print(label)
   if err != (label,):
       err = (label,)
   else:
       acc += 1

   acc_list.append(acc/count)
   print(acc/count)

   bn.update(time, sensor_inputs, err, ITER)
   ITER +=1

print(acc)


#acc_list.append(acc)
    #print(acc_list)
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_facecolor((0.91, 0.90, 0.90))
ax.plot(np.linspace(1,len(acc_list),len(acc_list)), acc_list, label= "Model Accuracy")
ax.legend()
ax.grid()

ax.set_xlabel("Iterations")
ax.set_ylabel("Accuracy")
ax.set_title("Accuracy throughout Iterations")
plt.show()

