import matplotlib.pyplot as plt
import numpy as np
from BayesSiren import BNSIREN
from BayesLSTM import BNLSTM
import random
from itertools import chain, combinations
from copy import deepcopy

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def normal_func(sensor):
    def normal_S1(x):
        return np.sin(x)
    def normal_S2(x):
        return np.cos(x)

    if sensor == "S1":
        return normal_S1
    if sensor == "S2":
        return normal_S2


def error_func(err, theta, theta2):
    def error1(x):
        return theta * np.cos(theta2 * x)

    def error2(x):
        return theta * np.sin(theta2 * x)

    def error3(x):
        return theta * np.cos(theta2 * x)

    if err == "Error1":
        return error1
    if err == "Error2":
        return error2
    if err == "Error3":
        return error3


def actual_error_func(err, theta, theta2):
    def error1(x):
        return theta * np.cos(theta2 * x)

    def error2(x):
        return theta * np.sin(theta2 * x)

    def error3(x):
        return theta * np.cos(theta2 * x)

    if err == "Error1":
        return error1
    if err == "Error2":
        return error2
    if err == "Error3":
        return error3

def generate_sensor_inputs(errors, actual_error_list, normal_func, sensors, time_frame, noise):
    sensor_input = {}
    for i in sensors:
        sensor_input[i] = 0.0

    for sensor,error_tuple in errors.items():
        if error_tuple == ():
            sensor_input[sensor] += normal_func(sensor)(time_frame)
            continue
        for err in error_tuple:
            sensor_input[sensor] += actual_error_list[err](time_frame)

    for i in sensors:
        sensor_input[i] += noise

    return sensor_input


def plotting(x, error_val_, sensor_inputs):
    plt.clf()
    plt.plot(x, error_val_["S1"], color="red", label="Estimated Line")
    plt.plot(x, sensor_inputs["S1"], color="green", label="Actual Line")
    plt.xlabel("Timestamp")
    plt.ylabel("Line Val")
    plt.title("Comparison between Estimated and Actual")
    plt.legend()
    plt.grid()
    plt.show()

def main():
    mse_list = []
    index = 0
    x_param = [0.4,0.6,0.8,1.0,1.2,1.4,1.6]

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_facecolor((0.91,0.90,0.90))
    accuracy_list = []
    sensors = ["S1", "S2"]



    for j in x_param:

        # initilalize the BN(with expert's guesses)
        #####################################################Change the exepert guesses ################################
        bn = BNSIREN(sensor_list=sensors, errors={"S1": ["Error1", "Error2"], "S2": ["Error1", "Error3"]},
                     prior_distributions={"Error1": error_func("Error1", 2, 5),
                                          "Error2": error_func("Error2", j, 1.5),
                                          "Error3": error_func("Error3", 0.6, 0.5)}, sigmas={"S1": 1, "S2": 1, "S3": 1})
        # find the errors
        errors = bn.bayesian_net_possibilities(np.array([1, 1]))[1:]

        # get the actual errors
        actual_errors = {"Error1": actual_error_func("Error1", 1.5, 5), "Error2": actual_error_func("Error2", 0.8, 1.5),
                         "Error3": actual_error_func("Error3", 0.3, 0.5)}

        error_overall_list = list(powerset(["Error1", "Error2", "Error3"]))[1:]
        for i in range(len(error_overall_list)):
            error_overall_list[i] = tuple(sorted(error_overall_list[i]))

        acc = {}
        overall_trials ={}
        ITER = 0
        mse_sub = {}

        for i in error_overall_list:
            mse_sub[i]= []
            acc[i] = 0
            overall_trials[i] = 0


        while True:

            ##################################################Change the batch size ###################################
            x = np.linspace(start=0, stop=20, num=20)

            ###################################################Change the noise of the input##########################
            noise = np.random.normal(0,0.01,20)

            #error is randomly chosen
            error = random.choices(errors, weights = [0.1,0.1,0.1,0.4, 0.1,0.1,0.1])[0]

            #generate random errors
            sensor_inputs = generate_sensor_inputs(error, actual_errors, normal_func, sensors, x, noise)
            error = tuple(sorted(set(error["S1"] + error["S2"])))
            #MSE is the differences between observed samples and true samples
            err, probs, mse, error_val_ = bn.bayesian_calculation_update(sensor_inputs, x, ITER)

            if err == ('Error2',):
                plotting(x,error_val_, sensor_inputs)
            overall_trials[error] += 1

            err = tuple(sorted(err))

            #calculate the general accuracy
            if err == error:
                acc[err] += 1

            print(err,error)


            #Append the mse to the mse list according to the error types (notice we only observe sensor 1)
            mse_sub[err].append(mse["S1"])

            #calculate the mse of error 2
            if err == ('Error2',):
                if len(mse_sub[err]) == 25:
                    break

            #in case we need experts in the loop!!!!
            # err = error

            bn.update(x, sensor_inputs, err, ITER)
            ITER +=1

        accuracy_list.append(acc)
        np.random.seed(index)
        color_arr = np.random.rand(3)
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++Accuracy collected+++++++++++++++++++++++++++++++++++++++++++++")
        print(acc)
        print(overall_trials)
        #ax.plot(list(range(len(mse_sub))), mse_sub, c=(color_arr[0],color_arr[1],color_arr[2]), label= "Batch Size: " + str(j))
        mse_list.append(mse_sub[('Error2',)])

    print(accuracy_list)
    #ax.legend()
    #ax.grid()

    #ax.set_xlabel("Iterations")
    #ax.set_ylim(bottom=0, top = 0.15)
    #ax.set_ylabel("MSE between actual signal and generated signal")
    #ax.set_title("MSE Convergence VS Observations Batch Size")
   # plt.show()

    #np.save("LSTM_expert_frequency.npy", np.array(mse_list))

main()