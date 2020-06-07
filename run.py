'''
This file is the main train program of the project
'''

from __future__ import unicode_literals, print_function, division
from TaggedGraphEmbedding.data_procesing import *
from TaggedGraphEmbedding.TGEmodeling import *

from utils import *
import torch

from model_tools import trainTGE, evaluate_tge, model_save
from model import EncoderRNN, AttnDecoderRNN, EmbedderRNN
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

operation = str(sys.argv[1])  # dataset

train_batch_size = 64
train_epoch = 20
num_samples = 100  # Number of samples to train on.
num_training = int(0.8 * num_samples)
num_test = int(0.1 * num_samples)
num_valid = num_samples - num_training - num_test
K = 2  # hop size
max_code_diclen = 200
max_des_diclen = 200
max_node_diclen = 200
is_beamsearch = True
beam_num = 3


if "HS" in operation:
    ds_name = 'HS'
    describe = read_dataset('./data/hs.in', num_samples)
    program = read_dataset('./data/hs.out', num_samples)
elif "MTG" in operation:
    ds_name = 'MTG'
    describe = read_dataset('./data/magic.in', num_samples)
    program = read_dataset('./data/magic.out', num_samples)
elif "EJDT" in operation:  # EJDT
    ds_name = 'EJDT'
    dataset = load_dataset('./data/EJDT.json', num_samples)  # list
    program, describe = file_dev(dataset)
else:
    print("Dataset not exist, please check the input. ")
    exit()

# Separating the original dataset into program & describe two parts
write(describe, './data/'+ds_name+'describe.txt', ds_name)  # write describes as a txt file
write(program, './data/'+ds_name+'program.txt', ds_name)  # write describes as a txt file

# Conclude the Signature information from program
method = conclude_signature(program)
inputNL = load_data('./data/'+ds_name+'describe.txt')
outputCode = load_code_data(target_path='./data/'+ds_name+'program.txt')

print('\n\ngraph establishing...')
method_list, call_relations = get_calling_relationship([i.replace('\n', '$') for i in outputCode])
# call_relations = padding_method_list(call_relations, method_list)
node_list, front_call, behind_call = build_instant_neighbour(call_relations)
front_call_dict, behind_call_dict = build_instant_neighbour_dict(node_list, front_call, behind_call)

source_batch, max_source_len, describe_dic_i2w, describe_dic_w2i, _ = add_padding(inputNL, is_code=False)
target_batch, max_target_len, code_dic_i2w, code_dic_w2i, _ = add_padding(outputCode, is_code=True)

node_list_onehot_dict = build_node_onehot_dict(code_dic_w2i, method_list)

# node_num = len(inputNL)
# max_col = len(input_onehot[0])
# code_length = len(input_onehot[0][0])

with torch.no_grad():
    for i in range(len(source_batch)):
        while 0 in source_batch[i]:
            source_batch[i].pop(-1)
        source_batch[i] = [1] + source_batch[i] + [2]
        source_batch[i] = torch.from_numpy(np.array(source_batch[i], dtype=np.int64)).to(device).view(-1, 1)

    for i in range(len(target_batch)):
        while 0 in target_batch[i]:
            target_batch[i].pop(-1)
        target_batch[i] = [1] + target_batch[i] + [2]
        target_batch[i] = torch.from_numpy(np.array(target_batch[i], dtype=np.int64)).to(device).view(-1, 1)

hidden_size = 256
MAX_LENGTH = max(max_source_len, max_target_len)

encoder = EncoderRNN(len(describe_dic_i2w), hidden_size).to(device)
embedder = EmbedderRNN(len(code_dic_i2w), len(code_dic_i2w), dropout=0.1).to(device)
attn_decoder = AttnDecoderRNN(hidden_size, len(code_dic_i2w), dropout_p=0.1,
                              max_length=max(max_source_len, max_target_len)).to(device)

node_onehot_t = [[]]  # h
node_onehot_t[0] = node_list_onehot_dict
for i in range(K):
    node_onehot_t.append(node_list_onehot_dict)

# print(len(node_onehot_t[0][method_list[0]]))
# exit()

# encoder_outputs, encoder_hidden = encoder(source_batch[0][0], encoder.initHidden())
# decoder_input = torch.tensor(node_onehot_t[0][method_list[0]], dtype=torch.int64).to(device)
# t = torch.tensor(node_onehot_t[0][method_list[0]], dtype=torch.float32).view(1, 1, -1).to(device)
# print(t)
#
# decoder_input = embedder(t).view(1, -1)
# # print(decoder_input)
# # print(decoder_input.shape)
# mm = torch.zeros(len(code_dic_i2w))
# mm[source_batch[0][0]] = 1
# print(mm.shape)
# mm = mm.view(1, -1)
# print(mm.shape)
# # print(mm)
#
# decoder_input = torch.tensor(mm, dtype=torch.int64).to(device)
# # decoder_input = torch.tensor(decoder_input, dtype=torch.double).to(device)
#
# deoutput, _, __ = attn_decoder(decoder_input, encoder_hidden, encoder_outputs)
# print(deoutput)
# # print(deoutput.shape)
# exit()



# embedder_input = torch.cat((node_onehot_t[0][method_list[0]], node_onehot_t[0][method_list[0]], node_onehot_t[0][method_list[0]]), dim=0).view(-1, len(code_dic_i2w))
# embedder_output = embedder(embedder_input.to(device))
# print(embedder_input.shape)
# print(len(node_onehot_t[0][method_list[0]]))
# print(len(embedder_output[-1]))
# print(embedder_output[-1].shape)
# print(embedder_output.shape)
# exit()

method_list_index = []
for m in method_list:
    method_list_index.append(code_dic_w2i[m])


training_source = source_batch[:num_training]
training_target = target_batch[:num_training]

test_source = source_batch[num_training:num_training + num_test]
test_target = target_batch[num_training:num_training + num_test]

valid_source = source_batch[-num_valid:]
valid_target = target_batch[-num_valid:]


