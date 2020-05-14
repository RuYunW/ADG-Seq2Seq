import json
import re
from gensim import corpora
import numpy as np
import os
import random
import itertools
from tqdm import tqdm


# read json file
def load_dataset(path, num_sample):
    with open(path, 'r') as f:
        datas = f.readlines()
    data = []
    print('data loading...')
    for d in datas:
        if len(str(d)) <= 2000:
            data.append(d)
        if len(data) == num_sample:
            break
    return data  # list


def read_dataset(path, num_samples):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
        data.append(line)
        if len(data) >= num_samples:
            break
    return data


# Separating the original dataset into program & describe two parts
def file_dev(data):
    program = []
    describe = []
    for i in data:
        prog = json.loads(i)['code'].replace('\n', '$')
        des = json.loads(i)['nl']
        program.append(prog)
        describe.append(des)
    return program, describe


# write describes as a txt file
def write(data, filename, ds_name):
    # before write file, make sure if the file exists, if dose, delete it
    if os.path.exists(filename):
        os.remove(filename)
    # open the file with writing, if file not exists, new one
    file_write_obj = open(filename, 'w')
    for var in data:
        file_write_obj.writelines(var)
        if ds_name == 'EJDT':
            file_write_obj.write('\n')
    file_write_obj.close()


# Conclude the Signature information from program (method)
# as the form of"public class AgelessEntity extends CardImpl {"
def conclude_signature(data):
    lines = data
    temp = []
    for i in lines:
        method_line = i.split('$')[0]
        temp.append(method_line)
    temp = list(filter(None, temp))
    method = []
    for i in temp:
        if len(str(i)) >= 2:
            method.append(i)
    return method


# clean method
# separating it into input. output, method_name
# return the popped_list to remember which item has been dropped
def generate_io(data):
    lines = data
    inlist = []
    outlist = []
    method_list = []
    t = 0

    for line in lines:
        method = str(line).replace('public ', '').replace('static ', '').replace('@Override ', '').replace('private ',
                                                                                                           '').replace(
            '@Deprecated ', '').replace('final ', '').replace('\n', '').replace('@deprecated ', '').replace(
            'protected ', '').replace('@ShortType ', '').replace('@VisibleForTesting ', '')
        method = re.sub('@(.*) ', '', method)
        input = re.findall(r' .*?\((.*?)\){', method)
        output = re.findall(r'(.*?) .*?\(.*?\){', method)
        input_class = []
        for i in str(input).split(','):
            input_class.append(i.split(' ')[0].replace('[', '').replace(']', '').replace('\'', ''))
        inlist.append(input_class)
        outlist.append(output)
        method_name = re.findall(r'.*? (.*?)\(.*?\)', method)
        if len(method_name) > 0:
            method_name[0] += str(t)
        t += 1
        method_list.append(method_name)

    # drop empty
    i = 0
    pop_list = []  # to store the index of popped item
    while i < len(method_list):
        if len(inlist[i]) == 0 or len(outlist[i]) == 0 or len(method_list[i]) == 0:
            method_list.pop(i)
            inlist.pop(i)
            outlist.pop(i)
            pop_list.append(i)
            i -= 1
        i += 1

    return inlist, outlist, method_list, pop_list


# reform dataset
# cited_paper_id \t citing_paper_id
def cited2citing(input, output, method, save_path):
    row = len(method)
    # list 2 dic
    dic = corpora.Dictionary(method)
    dic.save_as_text(save_path)
    dic_set = dic.token2id

    final = []
    for i in tqdm(range(row), desc='Data Processing', ncols=100):
        for ins in input[i]:
            if ins == 'int' or ins == 'char' or ins == 'long' or ins == 'float' or ins == 'double' or ins == 'boolean' \
                    or ins == "String" or ins == 'Object' or ins == 'byte':
                continue
            for j in range(row):
                # if input appeared in output
                if ins in output[j]:
                    cited_name = method[j][0]
                    citing_name = method[i][0]
                    cited_id = dic_set[cited_name]
                    citing_id = dic_set[citing_name]
                    final.append(str(cited_id) + '\t' + str(citing_id) + '\n')
    return final


def write_cited(data, filename):
    # before writing, make sure if the file exists, if dose, delete it
    if os.path.exists(filename):
        os.remove(filename)

    # open the file with write, if file not exists, new one
    file_write_obj = open(filename, 'w')
    for var in data:
        file_write_obj.writelines(var)
    file_write_obj.close()


# reform the data
# node_index \t onehot \t class
def content_file_generation(node_onehot_code, cited_path, dic_path, content_path):
    with open(cited_path, 'r') as f:
        lines = f.readlines()
        row = len(lines)
    node_list = []

    with open(dic_path, 'r') as f_useless:
        lines2 = f_useless.readlines()[1:]
        number = len(lines2)

    node = []  # temp list to store order set
    for i in tqdm(range(number), desc='Graph Establishing', ncols=100):  # dic length
        for j in range(row):  # cited length
            # traver front node
            if str(lines2[i]).split('\t')[0] == str(lines[j].split('\t')[1].replace('\n', '')):
                node.append(lines[j].split('\t')[0].replace('\n', ''))
                if str(i + 1) != str(lines[j + 1].split('\t')[1]):  # the cited file is sorted with citing index
                    break  # if the latter index has been scanned not match, this node traversing finished
        node.append(lines2[i].split('\t')[0])
        # traverse behind node
        for j in range(row):
            if str(lines2[i]).split('\t')[0] == str(lines[j].split('\t')[0]):
                node.append(lines[j].split('\t')[1].replace('\n', ''))

        node_list.append(node)
        node = []
    one_hot = node_onehot_code

    # before writing, make sure if the file exists, if dose, delete it

    if os.path.exists(content_path):
        os.remove(content_path)

    # print(len(node_list))
    # open the file with write, if file not exists, new one
    file_write_obj = open(content_path, 'w')
    isolate_list = []
    for var in range(number):
        if len(node_list[var]) == 1:  # ignore the isolated node
            isolate_list.append(var)  # append index to list, in order to ignore its matching NL in main program
            continue
        else:
            file_write_obj.write(str(lines2[var]).split('\t')[0] + "\t")  # add the node dic index
            for i in one_hot[var]:
                file_write_obj.writelines(str(int(i)) + "\t")  # add the one-hot code of node
            file_write_obj.write(
                str(random.randint(1, 5)) + '\n')  # add the class of node, temporarily generate randomly
    file_write_obj.close()
    return isolate_list


def get_node_onehot(node_input, node_output):
    class_set = set(list(itertools.chain.from_iterable(node_input + node_output)))  # 将list变为一维，同时转换为set去重
    node_token_index = dict([(cls, i) for i, cls in enumerate(class_set)])
    node_onehot_code = np.zeros(
        (len(node_input), len(class_set)), dtype='float16')

    for i in range(len(node_input)):
        for j in range(len(node_input[i])):
            if node_input[i][j] in node_token_index:
                node_onehot_code[i, node_token_index[node_input[i][j]]] = 1

    return node_onehot_code
