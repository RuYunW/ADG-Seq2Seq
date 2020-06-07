import numpy as np


def load_data(cites_path, content_path):
    with open(content_path, 'r') as f1:
        lines = f1.readlines()

    x = []
    for i in range(len(lines)):
        x.append(lines[i].split('\t')[0:-1])
    code_length = len(lines[0].split('\t')[1:-1])
    cites = []
    with open(cites_path, 'r') as f_cites:
        lines = f_cites.readlines()
    for line in lines:
        cites.append(line.replace('\n', '').split('\t'))

    content = []
    with open(content_path, 'r') as f_co:
        lines = f_co.readlines()
    for line in lines:
        content.append(line.replace('\n', '').split('\t'))

    class_set = set(np.array(content[:])[:, -1])

    return x, cites, content, class_set, code_length


def front_node_garthing(x, cites, content, class_set):
    temp_frontnode = []
    temp_frontcode = []
    front_node = []
    front_code = []
    temp = []

    for node_id in np.array(x)[:, 0]:  # the 1st col of content
        for line in cites:
            if line[1] == node_id:  # target node==current node
                temp_frontnode.append(line[0])  # add current node pre-node
        # find all pre-nodes's temp_frontnode -- matching x

        for cls in class_set:
            for line in content:
                if line[0] in temp_frontnode and line[-1] == cls:
                    temp.append(list(map(int,line[1:-1])))
            # store one of node classes' all codes into temp
            temp_frontcode.append(temp)
            temp = []

        front_node.append(temp_frontnode)
        front_code.append(temp_frontcode)

        temp_frontnode = []
        temp_frontcode = []

    return front_code


def ave(front_code):
    ave_code = []
    ___ = []
    for i in front_code:  # each node
        for j in i:  # each class
            # ___ is used to save one class ave value
            if len(j) >= 1:
                ___.append((np.sum([k for k in j], axis=0)/len(j)).tolist())
        # each class ave has been calculated, saving in ___
        ave_code.append(___)
        ___ = []

    return ave_code


def behind_node_garthing(x,cites,content,class_set):
    temp_behindnode = []
    temp_behindcode = []
    behind_node = []
    behind_code = []
    temp = []

    for node_id in np.array(x)[:, 0]:  # content第一列顺序
        for line in cites:
            if line[0] == node_id:
                temp_behindnode.append(line[1])
        # find all pre-nodes' temp_frontnode

        for cls in class_set:
            for line in content:
                if line[0] in temp_behindnode and line[-1] == cls:
                    temp.append(list(map(int,line[1:-1])))
            # store all class of one of node class's code into temp
            temp_behindcode.append(temp)
            temp = []

        behind_node.append(temp_behindnode)
        behind_code.append(temp_behindcode)
        # pure_code += temp_frontcode[1:-1]

        temp_behindnode = []
        temp_behindcode = []

    return behind_code


def h_front_cal(x,cites,content,class_set,h_front):
    temp_frontnode = []
    temp_frontcode = []
    front_node = []
    front_code = []
    temp = []
    # pure_code = []

    for node_id in np.array(x)[:, 0]:  # he 1st col of content
        for line in cites:
            if line[1] == node_id:
                temp_frontnode.append(line[0])
        # find all pre-nodes' temp_frontnode

        for cls in class_set:
            for line in content:
                counter = 0
                if line[0] in temp_frontnode and line[-1] == cls:
                    temp.append(h_front[counter])
                counter += 1
            # store all class of one of node class's code into temp
            temp_frontcode.append(temp)
            temp = []

        front_node.append(temp_frontnode)
        front_code.append(temp_frontcode)
        # pure_code += temp_frontcode[1:-1]

        temp_frontnode = []
        temp_frontcode = []

    return front_code


def summ(h,code_length):
    __ = []
    for i in h:
        if len(i) >= 1:
            __.append(np.sum([ j for j in i],axis=0).tolist())
        else:
            __.append([0 for _ in range(code_length)])  # padding with 0
    # sum each vir_class code within one
    return __


def h_behind_cal(x,cites,content,class_set,h_behind):
    temp_behindnode = []
    temp_behindcode = []
    behind_node = []
    behind_code = []
    temp = []
    # pure_code = []

    for node_id in np.array(x)[:, 0]:  # content第一列顺序
        for line in cites:
            if line[0] == node_id:
                temp_behindnode.append(line[1])
        # find all pre-nodes' temp_frontnode
        for cls in class_set:
            for line in content:
                counter = 0
                if line[0] in temp_behindnode and line[-1] == cls:
                    temp.append(h_behind[counter])
                counter += 1
            # store all class of one of node class's code into temp
            temp_behindcode.append(temp)
            temp = []

        behind_node.append(temp_behindnode)
        behind_code.append(temp_behindcode)
        # pure_code += temp_frontcode[1:-1]

        temp_behindnode = []
        temp_behindcode = []

    return behind_code




