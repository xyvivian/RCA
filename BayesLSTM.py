import numpy as np
from itertools import chain, combinations
from scipy.stats import norm
from sklearn.metrics import mean_squared_error
import collections
from copy import deepcopy
import seaborn as sns
from collections import OrderedDict

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import skimage
import matplotlib.pyplot as plt

def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int'''
    tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid

def convert_numpy_to_model_input(ts):
    """
    Convert a numpy of time steps into suitable model input for bayesian
    :param ts:
    :return:
    """
    sysout = np.sin(ts)
    nor_signal = Signal_Data(ts, sysout)
    dataloader_signal = DataLoader(nor_signal, shuffle=True, batch_size=1, pin_memory=True, num_workers=0)
    model_input, _ = next(iter(dataloader_signal))
    model_input =  model_input.cuda()
    return model_input


class Signal_Data(torch.utils.data.Dataset):
    def __init__(self, ts, inout):
        self.data = inout
        self.data = self.data.astype(np.float32)
        self.timepoints = get_mgrid(len(self.data), 1)

    def get_num_samples(self):
        return self.timepoints.shape[0]

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        amplitude = self.data
        scale = np.max(np.abs(amplitude))
        amplitude = (amplitude / scale)
        amplitude = torch.Tensor(amplitude).view(-1, 1)
        return self.timepoints, amplitude


# Here we define our model as a class
class LSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, batch_size, output_dim, num_layers=3):
        self.input_size = input_dim
        self.hidden_size = hidden_dim
        self.layers_size = num_layers
        self.output_size = output_dim
        self.batch_size = batch_size

        super(LSTM, self).__init__()
        # Define the LSTM layer
        self.lstm = self.model = nn.LSTM(self.input_size, self.hidden_size, self.layers_size, batch_first=True)
        # Define the output layer
        self.linear = nn.Linear(self.hidden_size, self.output_size)

    def init_hidden(self):
        # This is what we'll initialise our hidden state as
        return (torch.zeros(self.layers_size, self.batch_size, self.hidden_size),
                torch.zeros(self.layers_size, self.batch_size, self.hidden_size))

    def forward(self, inputs, current_batch):
        # Forward pass through LSTM layer
        # shape of lstm_out: [input_size, batch_size, hidden_dim]
        # shape of self.hidden: (a, b), where a and b both
        # have shape (num_layers, batch_size, hidden_dim).
        lstm_out, self.hidden = self.lstm(inputs.view(current_batch, inputs.shape[1], inputs.shape[2]))
        # Only take the output from the final timetep
        # Can pass on the entirety of lstm_out to the next layer if it is a seq2seq prediction
        y_pred = self.linear(lstm_out)
        return y_pred

    def loss_fn(self, output, target):
        loss = torch.mean((output.double() - target.double()) ** 2)
        return loss



def train_LSTM(dataloader, signal_LSTM = 1):
    model_input, ground_truth = next(iter(dataloader))
    model_input, ground_truth = model_input.cuda(), ground_truth.cuda()

    if signal_LSTM == 1:
        signal_LSTM = LSTM(input_dim = 1, hidden_dim = 256, batch_size = 5, output_dim = 1)
        signal_LSTM.cuda()


    total_steps = 500
    steps_til_summary = 200

    optim = torch.optim.Adam(lr=1e-4, params=signal_LSTM.parameters())

    for step in range(total_steps):
        model_output = signal_LSTM.forward(model_input,1)
        loss = F.mse_loss(model_output, ground_truth)

        if not step % steps_til_summary:
            print("Step %d, Total loss %0.6f" % (step, loss))

            #fig, axes = plt.subplots(1, 2)
            #axes[0].plot(model_input.squeeze().detach().cpu().numpy(), model_output.squeeze().detach().cpu().numpy())
            #axes[1].plot(model_input.squeeze().detach().cpu().numpy(), ground_truth.squeeze().detach().cpu().numpy())
            #plt.show()

        optim.zero_grad()
        loss.backward()
        optim.step()

    return signal_LSTM


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


def normal_func(sensor):
    def normal_S1(x):
        return np.sin(x)
    def normal_S2(x):
        return np.cos(x)

    def normal_S3(x):
        return np.cos(1.1*x)

    if sensor == "S1":
        return normal_S1

    if sensor == "S2":
        return normal_S2

    if sensor == "S3":
        return normal_S3


def error_func(err,theta):
    def error1(x):
        return theta* np.sin(x)
    def error2(x):
        return theta* np.cos(x)
    def error3(x):
        return theta* np.sin(10*x)
    def error4(x):
        return theta* np.cos(5*x)

    if err == "Error1":
        return error1
    if err == "Error2":
        return error2
    if err == "Error3":
        return error3
    if err == "Error4":
        return error4


def composite_err_func(err_list, theta_list, time_frame):
    err_val = []
    for err,theta in zip(err_list,theta_list):
         err_val.append(error_func(err,theta)(time_frame))
    return np.sum(err_val, axis=0)



class BN():
    def __init__(self, sensor_list, errors, prior_distributions, sigmas):
        # sensor_list: how many sensors
        # errors: the errors associated with the sensors,
        # SHOULD BE {"S1": tuple[ERROR1, ERROR2, ..], "S2": tuple[ERROR1,ERROR2..]}
        # prior_distribution: expert's guess on the distributions SHOULD BE {"Error1": -1COS(5T),...}
        # sigmas: variances of gaussian
        # sensor_error_list: experts' guesses with the sensors
        # stored_data: the data stored with the particular error
        self.sensor_list = sensor_list
        self.errors = errors
        self.sensor_size = len(self.sensor_list)
        self.sigmas = sigmas
        self.sensor_error_list = []
        self.LSTM_list = {}
        self.stored_data = {}
        self.errors_list = {}
        self.errors_name_list = {}
        for sensor in sensor_list:
            self.LSTM_list[sensor] = {}
        self.prior_distributions = prior_distributions



    def error_generation(self,error_list, error_name_list):
        error_combination_list = list((powerset(error_list)))[1:]
        error_name_combination_list = list((powerset(error_name_list)))[1:]

        # Get all linear combinations of the experts' guess
        error_result_list = []
        for err in error_combination_list:
            err_arr = np.array(err)
            if err_arr.shape[0] > 1:
                err_arr = np.sum(err_arr, axis=0)
            if err_arr.shape[0] == 1:
                err_arr = err_arr.reshape((err_arr.shape[1],))
            error_result_list.append(err_arr)

        return error_result_list, error_name_combination_list


    def create_error_lists(self,time_frame):
        #"S1","S2","S3"... {Error1, }
        for sensor, error in self.errors.items():
            err_list = []
            for single_err in error:
                err_list.append(self.prior_distributions[single_err](time_frame))
            err_list, err_name = self.error_generation(err_list, list(error))
            err_list.insert(0, normal_func(sensor)(time_frame))
            err_name.insert(0, ())
            self.errors_list[sensor] = err_list
            self.errors_name_list[sensor] = err_name

    def error_calculation(self,error_list, sensor, x):
        model_input = convert_numpy_to_model_input(x)
        lstm_model = self.LSTM_list[sensor][error_list]
        print(lstm_model)
        output = lstm_model.forward(inputs= model_input, current_batch = 1)
        output = output.cpu().detach().numpy()
        output = output.reshape((output.shape[1],))
        past_data = self.stored_data[error_list][1][sensor]
        scale = np.max(np.abs(past_data))
        output = output * scale
        return output



    def get_max(self,my_list):
        import operator
        index, value = max(enumerate(my_list), key=operator.itemgetter(1))
        return index,value


    def find_most_probable_error(self,err_names, probs):
        ind, prob = self.get_max(probs)
        print(ind, prob)
        print(err_names[ind])
        return err_names[ind], prob


    def search_helper(self, err_prefix, err_name_prefix, errors, error_names):
        if sorted(err_prefix.keys()) == sorted(self.errors_list.keys()):
            errors.append(deepcopy(err_prefix))
            error_names.append(deepcopy(err_name_prefix))
            return
        else:
            visited_sensors = list(err_prefix.keys())
            total_sensors = list(self.errors_list.keys())
            remained_sensors = np.setdiff1d(total_sensors, visited_sensors)
            targeted_sensors = remained_sensors[0]

            for i in range(len(self.errors_list[targeted_sensors])):
                err_prefix[targeted_sensors] = self.errors_list[targeted_sensors][i]
                err_name_prefix[targeted_sensors] = self.errors_name_list[targeted_sensors][i]

                self.search_helper(err_prefix,err_name_prefix,errors,error_names)
                err_prefix.pop(targeted_sensors, self.errors_list[targeted_sensors][i])
                err_name_prefix.pop(targeted_sensors,self.errors_name_list[targeted_sensors][i])

    def train_LSTM(self,dataloader, error_term , sensor, ITER):
        if error_term not in self.LSTM_list[sensor].keys():
             signal_LSTM = train_LSTM(dataloader)
             self.LSTM_list[sensor][error_term] = signal_LSTM
        else:
             self.LSTM_list[sensor][error_term] = train_LSTM(dataloader, self.LSTM_list[sensor][error_term])


    def LSTM_update(self,sensor_inputs,time_frame, error_term,sensor, ITER):
        sensor_input = sensor_inputs[sensor]
        abnormal_signal = Signal_Data(time_frame, sensor_input)
        dataloader_signal = DataLoader(abnormal_signal, shuffle=False, batch_size=1, pin_memory=True, num_workers=0)
        self.train_LSTM(dataloader=dataloader_signal,error_term=error_term,sensor = sensor, ITER = ITER)


    def check_conflicts(self, errors, error_names):
        ret_err = []
        ret_err_name = []
        inv_err = collections.defaultdict(set)
        for k, v in self.errors.items():
            for item in v:
                inv_err[item].add(k)

        for i in range(len(error_names)):
            candidate = error_names[i]
            checked_error = set()
            wrong_config = False
            for sensor,error in candidate.items():
                for err in error:
                    if err in checked_error:
                        break
                    checked_error.add(err)
                    required_indices = list(inv_err[err])
                    wrong_config = any([ err not in candidate[i] for i in required_indices])
                    if wrong_config:
                        break
                if wrong_config:
                    break
            if not wrong_config:
                ret_err.append(errors[i])
                ret_err_name.append(error_names[i])
        return ret_err, ret_err_name



    # Baysian Calculation)
    # P(S2=y2|E3= 0,E4= 0)·P(S1=y1|E1= 1,E2= 0,E3= 0)· 1/ 16
    # Assuming each error occurs equally (so P(E1= 1)P(E2= 0)P(E3= 0)P(E4= 0) can be ignored )
    def bayesian_calculation_update(self, sensor_inputs,time_frame, ITER):
        self.create_error_lists(time_frame)
        err_names = []
        probs = []
        errors = []
        error_names = []
        self.search_helper({}, {}, errors, error_names)
        # need to make sure all the lists do not have conflicts
        errors, error_names = self.check_conflicts(errors, error_names)
        for i in range(len(errors)):
            err_name = tuple(set([item for sublist in error_names[i].values() for item in sublist]))
            err_val = errors[i]
            for key, value in self.LSTM_list.items():
                if err_name in value.keys():
                    data_size = self.stored_data[err_name][1][key].shape[0]
                    val = self.error_calculation(error_list = err_name, sensor=key, x=time_frame)
                    err_val[key] = (1 - 1 / np.log(data_size)) * val + (1 / np.log(data_size)) * err_val[key]
                    #plt.clf()
                    #plt.plot(time_frame, err_val[key], color= "red", label = "Estimated Line")
                    #plt.plot(time_frame, sensor_inputs[key], color = "green", label = "Actual Line")
                    #plt.legend()

                    #plt.show()
                    #plt.savefig("rca_plot/error_calculation" + str("_comparison") + str(ITER) + ".png")
            prob = 0.0
            for j in self.sensor_list:
                for i in range(len(sensor_inputs[j])):
                    gaussian_err = norm.pdf(sensor_inputs[j][i], loc=err_val[j][i], scale=self.sigmas[j])
                    prob += np.log(gaussian_err)

            err_names.append(err_name)
            probs.append(prob)

        error, pro = self.find_most_probable_error(err_names, probs)
        return error, pro, (err_names, probs)



    # Train a new MLE that better fits the data
    def update(self, time_frame, sensor_inputs,errors, ITER):
        self.update_data_map(time_frame,sensor_inputs,errors)
        #list of empty tuples
        error_list = {}
        for s in self.sensor_list:
            error_list[s] = (tuple())

        for err in errors:
            for key, value in self.errors.items():
                if err in value:
                    error_list[key] = error_list[key] + (err,)
        for sensor_key,err_value in error_list.items():
            if err_value != ():
                self.LSTM_update(sensor_inputs=sensor_inputs,time_frame=time_frame,error_term=errors,sensor = sensor_key, ITER= ITER)

    # update data_map
    def update_data_map(self,time_frame,sensor_inputs,error):
        if error in self.stored_data:
            old_time_frame = self.stored_data[error][0]
            old_sensors = self.stored_data[error][1]

            updated_time_frame = np.hstack((old_time_frame, time_frame))
            updated_sensor_inputs = deepcopy(sensor_inputs)
            for key,_ in updated_sensor_inputs.items():
                updated_sensor_inputs[key] = np.hstack((old_sensors[key], sensor_inputs[key]))

            self.stored_data[error] = (updated_time_frame, updated_sensor_inputs)
        else:
            self.stored_data[error] = (time_frame, sensor_inputs)



def main():
    ITER = 0
    x = np.linspace(start=0, stop=20, num=20)
    bn = BN(sensor_list=["S1", "S2"], errors={"S1": ["Error1", "Error2", "Error3"], "S2": ["Error1", "Error3"]},
                  prior_distributions={"Error1": error_func("Error1", theta=4), "Error2": error_func("Error2", theta=2),
                                       "Error3": error_func("Error3", theta=0.5)}, sigmas={"S1": 1, "S2": 1, "S3": 1})
    for iterations in range(10):
            sensor_inputs = {"S1": normal_func("S1")(x), "S2": -0.6* np.cos(5*x)}
            err, probs, _ = bn.bayesian_calculation_update(sensor_inputs, x, ITER)
            if err != ('Error3',):
                print("Wrong Estimation")
                err = ('Error3',)
            bn.update(x, sensor_inputs, err, ITER)
            ITER +=1


if __name__ == '__main__':
    main()

