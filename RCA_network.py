import numpy as np
import matplotlib.pyplot as plt
from itertools import chain, combinations
from scipy.stats import norm
from sklearn.metrics import mean_squared_error
import collections
from copy import deepcopy
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import itertools
from unittest.mock import patch
import unittest

class RCA_Network:
    def __init__(self):
        self.G = nx.DiGraph()
        self.fm_size = int(input("Enter Number of Failure Modes"))
        self.st_size = int(input("Enter Number of Sensors"))
        self.failure_modes = []
        self.sensors = []

        while (len(self.failure_modes) != self.fm_size):
            inp_fm = input(f"Enter failure mode name (String) {len(self.failure_modes) + 1}: ")
            inp_save = input(f"Save {inp_fm} ? (y/n): ")
            if inp_save == 'y':
                self.failure_modes.append(inp_fm)

        for i in range(self.fm_size):
            self.G.add_node(self.failure_modes[i])

        while (len(self.sensors) != self.st_size):
            inp_st = input(f"Enter sensor type name (String) {len(self.sensors) + 1}: ")
            inp_save = input(f"Save {inp_st} ? (y/n): ")
            if inp_save == 'y':
                self.sensors.append(inp_st)

        for i in range(self.st_size):
            self.G.add_node(self.sensors[i])

        self.adj_mat = nx.to_numpy_array(self.G)
        self.ftos_mat = np.max(self.adj_mat, axis=0)[self.fm_size:]
        self.all_sensor_good = np.min(self.ftos_mat)
        self.bad_sensor = -1
        if (self.all_sensor_good == 0):
            self.bad_sensor = np.argmin(self.ftos_mat)

        while (self.all_sensor_good == 0):
            print(f"Enter failure modes detected by sensor {self.sensors[self.bad_sensor]}")
            for j in range(self.fm_size):
                inp_fmv = input(f"Is failure mode {self.failure_modes[j]} detected by this sensor ? (y/n): ")
                if inp_fmv == 'y':
                    self.G.add_edge(self.failure_modes[j], self.sensors[self.bad_sensor])
            self.adj_mat = nx.to_numpy_array(self.G)
            self.ftos_mat = np.max(self.adj_mat, axis=0)[self.fm_size:]
            self.all_sensor_good = np.min(self.ftos_mat)
            if (self.all_sensor_good == 0):
                self.bad_sensor = np.argmin(self.ftos_mat)
            # print(self.all_sensor_good)
            # print(self.ftos_mat)
            # print(self.adj_mat)
            # print(self.bad_sensor, np.argmin(self.ftos_mat), self.fm_size)

        print('Bayesian Network Complete')

    def get_graph(self):
        return self.G

    def get_adj_matrix(self):
        return nx.to_numpy_array(self.G)

    def plot_graph(self):
        colour_map = ['blue' for f in self.failure_modes] + ['yellow' for s in self.sensors]
        size_map = [2000 for f in self.failure_modes] + [1000 for s in self.sensors]
        plt.figure(figsize=(10, 10))
        pos = nx.bipartite_layout(self.G, self.failure_modes, align='horizontal')
        nx.draw_networkx_nodes(self.G, pos, node_color=colour_map, node_size=size_map, alpha=0.4, edgecolors='black',
                               linewidths=2)
        nx.draw_networkx_labels(self.G, pos, font_weight='bold')
        nx.draw_networkx_edges(self.G, pos, arrowsize=25, connectionstyle="arc3,rad=-0.1", min_source_margin=22,
                               min_target_margin=16)
        plt.savefig("Network.png")

    def get_detected_dict(self):
        RCAG = self.G
        detected_dic = {}
        for nd in list(RCAG.nodes())[fs:]:
            detected_dic.update({nd: [x[0] for x in RCAG.in_edges(nd)]})
        return detected_dic


def test_rca_network(fs, ss, cs):
    inputs = []
    finputs = []
    sinputs = []
    inputs.append(str(fs))
    inputs.append(str(ss))
    for i in range(fs):
        inputs.append('Error' + str(i + 1))
        finputs.append('Error' + str(i + 1))
        inputs.append('y')

    for i in range(ss):
        inputs.append('S' + str(i + 1))
        sinputs.append('S' + str(i + 1))
        inputs.append('y')

    all_edges = list(itertools.product(sinputs, finputs))
    B = nx.algorithms.bipartite.generators.gnmk_random_graph(ss, fs, cs, directed=True)
    relabel_dict = {k: v for k, v in enumerate(sinputs + finputs)}
    B = nx.relabel_nodes(B, relabel_dict)
    cur_edges = list(B.edges)

    print(all_edges, cur_edges)

    for ed in all_edges:
        if ed in cur_edges:
            inputs.append('y')
        else:
            inputs.append('n')

    # inputs = ['3', '2', 'F1', 'y', 'F2', 'y', 'F3', 'y', 'S1', 'y', 'S2', 'y', 'y', 'y', 'n', 'n', 'y', 'y']
    # print(inputs)
    with patch('builtins.input', side_effect=inputs):
        network = RCA_Network()
    return network



if __name__ == '__main__':
    fsnames = ['Error1', 'Error2', 'Error3']
    ssnames = ['S1', 'S2']
    fnames = fsnames
    snames = ssnames
    fs = 3
    ss = 2
    cs = 5
    network = test_rca_network(fs, ss, cs)
    network.plot_graph()


