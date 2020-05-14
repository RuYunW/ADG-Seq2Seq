'''
Tag-graph embedding generation algorithm
'''

from TaggedGraphEmbedding.TGEutils import *
from numpy import *
import re
from tqdm import tqdm
import torch


def run(cites_path, content_path):
    # Initialize
    x, cites, content, class_set, code_length = load_data(cites_path, content_path)

    order = []  # Create an empty list
    K = 2  # hops
    h = [[] for i in range(K+1)]  # temp matrices

    node_num = len(x[:])  # number of node
    # print(x[0][0])  # [[0,...,0],[...],...,[...]]

    h[0] += x[:]  # x is node's one-hot code itself
    y = (np.array(h[0])[:, [i for i in range(1, code_length+1)]]).tolist()  # drop the 1st col

    temp = []
    emb = []
    h_front = [[] for i in range(K+1)]
    h_behind = [[] for i in range(K+1)]
    ___ = ave(front_node_garthing(x, cites, content, class_set))
    max_col = max(len(j) for j in ___)
    zo = [0 for _ in range(code_length)]

    for i in range(len(___)):
        for j in range(max_col-len(___[i])):
            ___[i].append(zo)

    order_set = []
    M = len(x[:])
    #
    # # h_sum = sum(h_front)
    # for k in range(1,K+1):  # hop数  K
    #     h_front[k] = summ(ave(h_front_cal(x, cites, content, class_set,h_front[k-1])),code_length)
    #     h_behind[k] = summ(ave(h_behind_cal(x, cites, content, class_set,h_behind[k-1])),code_length)
    #     temp = []
    #     for m in range(M):
    #         temp.append(h_front[k][m]+h_behind[k][m])  # 拼接
    #     order_set.append(temp)

    trainX = np.array(___)
    trainY = np.array(x)[:, 1:].tolist()
    return trainX, trainY

def build_instant_neighbour_dict(node_list, front_call, behind_call):
    front_call_dict = {}
    behind_call_dict = {}
    for i in range(len(node_list)):
        front_call_dict[node_list[i]] = front_call[i]
        behind_call_dict[node_list[i]] = behind_call[i]
    return front_call_dict, behind_call_dict


def build_node_onehot_dict(code_dic_w2i, node_list):
    node_list_onehot = torch.zeros((len(node_list), len(code_dic_w2i)), dtype=torch.double)
    node_list_onehot_dict = {}
    for i in range(len(node_list)):
        node_list_onehot[i][code_dic_w2i[node_list[i]]] = 1
        node_list_onehot_dict[node_list[i]] = node_list_onehot[i]
    return node_list_onehot_dict


def build_instant_neighbour(call_relations):
    node_list = []
    front_call = []
    behind_call = []
    fcall = []
    for i in call_relations:
        node_list.append(i[0])
        behind_call.append(i[1:])

    for i in range(len(node_list)):
        for j in range(len(node_list)):
            if node_list[i] in call_relations[j][1:]:
                fcall.append(call_relations[j][0])
        front_call.append(fcall)
        fcall = []

    return node_list, front_call, behind_call


def get_calling_relationship(program):
    func_name = []
    call_relations = []
    for i in tqdm(program):
        call_relation = []
        Func_name = re.findall(' (\w+?)\(', i.split('$')[0])
        if len(Func_name) > 0:
            for Fname in Func_name:
                Fname = re.sub(r'[^\w]', ' ', Fname)
                if len(list(filter(None, Fname.split(' ')))) == 1 and not Fname.replace(' ', '').isdigit() and not 'if' == Fname.replace(' ', '') and not 'for' == Fname.replace(' ', '') and not 'while' == Fname.replace(' ', ''):
                    call_relation.append(Fname.replace(' ', ''))
                    func_name.append(Fname.replace(' ', ''))
        else:
            Fname = i.split('$')[0].split('(')[0]
            if "@" not in Fname:
                call_relation.append(Fname)
                func_name.append(Fname)
            else:
                Fname = re.findall(' (\w+?)\(', i.split('$')[0])
                call_relation.append(Fname[0])
                func_name.append(Fname[0])

        for sentence in i.split('$'):
            fnames = re.findall('\.(\w+?)\(', sentence)
            if len(fnames) > 0:
                for fname in fnames:
                    fname = re.sub(r'[^\w]', ' ', fname)
                    if len(list(filter(None, fname.split(' ')))) > 1 or fname.replace(' ', '').isdigit() or 'if' == fname.replace(' ', '') or 'for' == fname.replace(' ', '') or 'while' == fname.replace(' ',''):
                        continue
                    else:
                        func_name.append(fname.replace(' ', ''))
                        call_relation.append(fname.replace(' ', ''))
        call_relations.append(call_relation)
    return list(filter(None, list(set(func_name)))), call_relations


# def padding_method_list(call_relations, method_list):

