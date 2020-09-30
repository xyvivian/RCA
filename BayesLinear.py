import numpy as np
import matplotlib.pyplot as plt
from itertools import chain, combinations
from scipy.stats import norm
from sklearn.metrics import mean_squared_error
import collections
from copy import deepcopy
from scipy.stats import norm
import seaborn as sns


from itertools import chain, combinations
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
        # thetas: updated parameters
        # stored_data: the data stored with the particular error
        self.sensor_list = sensor_list
        self.errors = errors
        self.sensor_size = len(self.sensor_list)
        self.sigmas = sigmas
        self.sensor_error_list = []
        self.thetas = {}
        for sensor in sensor_list:
            self.thetas[sensor] = {}
        self.stored_data = {}
        self.errors_list = {}
        self.errors_name_list = {}
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

    def error_calculation(self,error_list, theta, x):
        err_val = []
        for err in error_list:
            err_val.append(error_func(err,theta)(x))
        return np.sum(err_val, axis=0)


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
    def bayesian_calculation_update(self, sensor_inputs,time_frame):
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
            for key, value in self.thetas.items():
                if err_name in value.keys():
                    data_size = self.stored_data[err_name][1][key].shape[0]
                    theta_prime = self.thetas[key][err_name]
                    val = self.error_calculation(error_names[i][key], theta_prime, time_frame)
                    err_val[key] = (1 - 1 / np.log(data_size)) * val + (1 / np.log(data_size)) * err_val[key]

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
    def update(self, time_frame, sensor_inputs,errors):
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
            theta_prime = 0.0
            if err_value != ():
                theta_prime = self.MLE_update(sensor_key,err_value, errors)
            if theta_prime != 0.0:
                self.thetas[sensor_key][errors] = theta_prime


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

    #MLE_update calculation
    def MLE_update(self,sensor_key, err_value, errors):
        y = []
        re_x, re_list = self.stored_data[errors]
        for err in err_value:
            y.append(error_func(err, 1)(re_x))
        y = np.sum(np.array(y), axis=0)
        theta_prime = np.mean(np.divide(re_list[sensor_key], y, out=np.zeros_like(re_list[sensor_key]), where=y != 0))
        return theta_prime


if __name__ == '__main__':
    x = np.linspace(start=0, stop=5, num=50)
    theta_S1 = []
    theta_S2 = []
    bn = BN(sensor_list=["S1", "S2", "S3"],
            errors={"S1": ["Error1", "Error2", "Error3"], "S2": ["Error3", "Error4"], "S3": ["Error1", "Error4"]},
            prior_distributions={"Error1": error_func("Error1", theta=4), "Error2": error_func("Error2", theta=2),
                                 "Error3": error_func("Error3", theta=1), "Error4": error_func("Error4", theta=-1)},
            sigmas={"S1": 1, "S2": 1, "S3": 1})

    count = 0
    for i in range(30):
        sensor_inputs = {"S1": error_func("Error3", theta=0.4)(x) + np.random.uniform(0, 0.1, 50), "S2": error_func("Error3", theta=0.3)(x) + np.random.uniform(0, 0.1, 50),
                     "S3": normal_func("S3")(x)}

        err, _,_ = bn.bayesian_calculation_update(sensor_inputs, x)
        bn.update(x, sensor_inputs, err)

        if err != ('Error3',):
            count += 1

        sensor_inputs = {"S1": error_func("Error2", theta=2)(x), "S2": normal_func("S2")(x),
                     "S3": normal_func("S3")(x)}
        err, _,_ =bn.bayesian_calculation_update(sensor_inputs, x)
        bn.update(x, sensor_inputs, err)

        theta_S1.append(bn.thetas["S1"])
        theta_S2.append(bn.thetas["S2"])

    theta_S1_E3 = [x[('Error3',)] for x in theta_S1]
    theta_S1_E2 = [x[('Error2',)] for x in theta_S1]
    theta_S2_E3 = [x[('Error3',)] for x in theta_S2]

    print(theta_S1_E3)
    print(theta_S2_E3)

    print(count)

    fig, ax = plt.subplots()
    sns.lineplot(x=np.arange(30), y=theta_S1_E3, label='Predicted Theta(1),Errror3',ax=ax)
    ax.legend(loc='upper right')

    sns.lineplot(x=np.arange(30), y=theta_S2_E3, label='Predicted Theta(2),Errror3', ax=ax)
    ax.legend(loc='upper right')

    sns.lineplot(x=np.arange(30), y=np.full((30,), 0.4), label='True Theta(1) Error3', ax=ax)
    sns.lineplot(x=np.arange(30), y=np.full((30,), 0.3), label='True Theta(2) Error3', ax=ax)

    ax.set_xlabel('iterations')
    ax.set_ylabel('predicted theta')
    ax.legend(loc='upper right')
    ax.set_title('Theta CHANGE')
    fig.show()













