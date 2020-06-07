from ADGraphEmbedding.ADGmodeling import run
import re
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


# enerate input and output，store as two list
def load_data(source_path):
    inputNL = []
    with open(source_path, 'r') as fs:
        lines = fs.readlines()
        for line in lines:
            inputNL.append(line)
    # input_onehot, output_onehot = run(cites_path, content_path)

    return inputNL


def load_code_data(target_path):
    outputCode = []
    with open(target_path, 'r') as ft:
        lines = ft.readlines()
        for line in lines:
            outputCode.append(line)
    return outputCode


# generate the dictionary of all words in doc
# w2i is word → index，i2w is index → word，
# both are matching
def make_vocab(docs, is_code=False):
    w2i = {"_PAD": 0, "_GO": 1, "_EOS": 2}
    i2w = {0: "_PAD", 1: "_GO", 2: "_EOS"}
    method_list = []
    if not is_code:
        for doc in docs:
            for w in doc.split(' '):
                if w not in w2i:
                    i2w[len(w2i)] = w
                    w2i[w] = len(w2i)

    else:  # code
        for doc in docs:
            for w in doc.split(' '):
                if w not in w2i:
                    i2w[len(w2i)] = w
                    w2i[w] = len(w2i)

    return w2i, i2w, method_list


def build_call_dict_onehot(code_dic_w2i, front_call_dict, behind_call_dict, node_list):
    front_call_dict_onehot = {}
    behind_call_dict_onehot = {}
    max_front_call_num = 0
    max_behind_call_num = 0
    for i in range(len(node_list)):
        if max_front_call_num < len(front_call_dict[node_list[i]]):
            max_front_call_num = len(front_call_dict[node_list[i]])
        if max_behind_call_num < len(behind_call_dict[node_list[i]]):
            max_behind_call_num = len(behind_call_dict[node_list[i]])

    for i in range(len(node_list)):
        front_call_dict_onehot[node_list[i]] = np.zeros((max_front_call_num, len(code_dic_w2i)), dtype='uint8')
        if len(front_call_dict[node_list[i]]) > 0:
            for fcall_index in range(len(front_call_dict[node_list[i]])):
                front_call_dict_onehot[node_list[i]][fcall_index][
                    code_dic_w2i[front_call_dict[node_list[i]][fcall_index]]] = 1

        behind_call_dict_onehot[node_list[i]] = np.zeros((max_behind_call_num, len(code_dic_w2i)), dtype='uint8')
        if len(behind_call_dict[node_list[i]]) > 0:
            for bcall_index in range(len(behind_call_dict[node_list[i]])):
                behind_call_dict_onehot[node_list[i]][bcall_index][
                    code_dic_w2i[behind_call_dict[node_list[i]][bcall_index]]] = 1

    return front_call_dict_onehot, behind_call_dict_onehot, max_front_call_num, max_behind_call_num


# transform the statement of doc into index list
# like [1, 11, 5, 14, 2, 0, 0, 0, 0, 0]
def doc_to_seq(docs):
    w2i = {"_PAD": 0, "_GO": 1, "_EOS": 2}
    i2w = {0: "_PAD", 1: "_GO", 2: "_EOS"}
    seqs = []
    for doc in docs:
        seq = []
        # doc = re.sub('[^\w]', ' ', doc)  # del char
        # for w in list(filter(None, doc.split(' '))):  # del null
        for w in doc.split(' '):
            if w not in w2i:
                i2w[len(w2i)] = w  # generate dictionary again
                w2i[w] = len(w2i)
            seq.append(w2i[w])
        seqs.append(seq)
    return seqs, w2i, i2w


# adding pad to input, change it into index mode
def add_padding(docs_source, is_code):
    clean_docs = []
    for doc in docs_source:
        doc = re.sub('[^\w]', ' ', doc)
        clean_docs.append(' '.join(list(filter(None, doc.split(' ')))))
    # generate dictionary
    w2i_source, i2w_source, method_list = make_vocab(clean_docs, is_code)

    # find max_length
    source_lens = [len(str(i).split(' ')) for i in clean_docs]
    max_source_len = max(source_lens)
    source_batch = []
    # padding
    for i in range(len(clean_docs)):
        source_seq = [w2i_source[w] for w in str(clean_docs[i]).split(' ')] + [w2i_source["_PAD"]] * (
                max_source_len - len(str(clean_docs[i]).split(' ')))
        source_batch.append(source_seq)

    return source_batch, max_source_len, i2w_source, w2i_source, method_list


def ignore_NL(inputNL, pop_list, isolate_list):
    for i in range(len(pop_list) - 1, -1, -1):
        inputNL.pop(pop_list[i])
    for i in range(len(isolate_list) - 1, -1, -1):
        inputNL.pop(isolate_list[i])
    return inputNL


def normalization(x):
    min = np.min(x)
    max = np.max(x)
    x = (x - min) / (max - min)
    return x


def reNorm(x, c, d):
    min = np.min(x)
    max = np.max(x)
    x = (x - min) * (d - c) / (max - min)
    return x


# draw figure
def draw_training_pic(history):
    plt.subplot(211)
    plt.title("Accuracy")
    plt.plot(history.history['accuracy'], color="g", label="Train")
    plt.plot(history.history["val_accuracy"], color="b", label="Validation")
    plt.legend(loc="best")

    plt.subplot(212)
    plt.title("Loss")
    plt.plot(history.history["loss"], color="g", label="Train")
    plt.plot(history.history["val_loss"], color="b", label="Validation")
    plt.legend(loc="best")

    plt.tight_layout()
    plt.show()


def id2onehot(ids, max_length, dic_len):
    one_hot = np.zeros((len(ids), max_length, dic_len), dtype='uint8')
    for i in range(len(ids)):  # item
        for j in range(max_length):  # word id
            one_hot[i][j][ids[i][j]] = 1
    return one_hot


def index2text(index_tensor_list, describe_dic_i2w):
    return_text = []
    for i in range(len(index_tensor_list)):
        return_text.append(describe_dic_i2w[index_tensor_list[i].item()])
    return return_text
