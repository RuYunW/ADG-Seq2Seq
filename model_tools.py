from __future__ import unicode_literals, print_function, division
import random
import torch
import torch.nn as nn
from torch import optim
import time
import math
import numpy as np

SOS_token = 1
EOS_token = 2
teacher_forcing_ratio = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def get_hop_orderset(k, m, node_onehot_t, behind_call_dict, front_call_dict, embedding_dim):
    # print(len(node_onehot_t))
    # with torch.no_grad
    b_onehot = []  # behind
    f_onehot = []  # front
    if m in list(behind_call_dict.keys()) and len(behind_call_dict[m]) > 0:
        for bcall in behind_call_dict[m]:
            b_onehot.append(np.array(node_onehot_t[k][bcall].cpu().detach().numpy(), dtype=np.float32))

    if m in list(front_call_dict.keys()) and len(front_call_dict[m]) > 0:
        for fcall in front_call_dict[m]:
            f_onehot.append(np.array(node_onehot_t[k][fcall].cpu().detach().numpy(), dtype=np.float32))

    b_onehot = torch.from_numpy(np.array(b_onehot, dtype=np.float32))
    # print(b_onehot)
    f_onehot = torch.from_numpy(np.array(f_onehot, dtype=np.float32))
    node_onehot_t[k][m] = torch.tensor(node_onehot_t[k][m], dtype=torch.float32).view(-1, embedding_dim)
    if len(b_onehot) >0 and len(f_onehot) >0:
        order_set_onehot = torch.cat((b_onehot, node_onehot_t[k][m], f_onehot), dim=0)
    elif len(b_onehot) >0 and len(f_onehot) == 0:
        # print(b_onehot.shape)
        # print(node_onehot_t[k][m].shape)
        order_set_onehot = torch.cat((b_onehot, node_onehot_t[k][m]), dim=0)
    elif len(f_onehot) >0 and len(b_onehot) == 0:
        order_set_onehot = torch.cat((node_onehot_t[k][m], f_onehot), dim=0)
    else:
        order_set_onehot = node_onehot_t[k][m]

    return order_set_onehot


def train_dfg(input_tensor, target_tensor, encoder, embedder, decoder,
              encoder_optimizer, embedder_optimizer, decoder_optimizer, criterion, max_length,
              method_list, node_list_onehot_dict, K,
              behind_call_dict, front_call_dict,
              node_onehot_t, code_dic_i2w):  # method_list

    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    embedder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)
    decoder_hidden = encoder_hidden

    # update all embeddings
    for k in range(K):  # hop
        for m in method_list:  # a
            embedder_inputs = get_hop_orderset(k, m, node_onehot_t, behind_call_dict, front_call_dict,
                                               len(code_dic_i2w)).to(device)
            embedder_inputs = embedder_inputs.view(1, -1, len(code_dic_i2w)).to(device)
            # print(embedder_inputs.shape)
            # exit()
            # embedder_inputs.view()
            embedder_output = embedder(embedder_inputs.to(device))
            node_onehot_t[k + 1][m] = embedder_output

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            token_index = target_tensor[di].item()
            token = code_dic_i2w[token_index]

            if token in method_list:  # replace
                embedder_inputs = node_onehot_t[K][token]
                embedder_output = embedder(embedder_inputs.view(1, 1, -1))
                decoder_input = embedder_output

            else:  # do not replace
                decoder_input = target_tensor[di]

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:  # EOS_token = 1
                break

    loss.backward()

    encoder_optimizer.step()
    embedder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length, node_onehot_t


def trainDFG(encoder, embedder, decoder, n_iters,
             training_inputs, training_outputs,
             method_list, node_list_onehot_dict, K,
             behind_call_dict, front_call_dict,
             node_onehot_t, code_dic_i2w, max_length,
             print_every=1000, learning_rate=0.01):
    start = time.time()
    print_loss_total = 0  # Reset every print_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    embedder_optimizer = optim.Adam(embedder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        # training_pair = training_pairs[iter - 1]
        input_tensor = training_inputs[iter - 1]
        target_tensor = training_outputs[iter - 1]

        # node_onehot_t = node_list_onehot_dict

        loss, node_onehot_t = train_dfg(input_tensor, target_tensor, encoder, embedder, decoder,
                                        encoder_optimizer, embedder_optimizer, decoder_optimizer, criterion,
                                        max_length=max_length,
                                        method_list=method_list, node_list_onehot_dict=node_list_onehot_dict, K=K,
                                        behind_call_dict=behind_call_dict, front_call_dict=front_call_dict,
                                        node_onehot_t=node_onehot_t, code_dic_i2w=code_dic_i2w)

        print_loss_total += loss
        # plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) loss: %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

    return node_onehot_t


def evaluate_dfg(encoder, embedder, decoder,
                 sentence, max_length, node_onehot_t, code_dic_i2w, method_list, K):
    with torch.no_grad():
        input_tensor = sentence
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()
        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('_EOS')
                break
            else:
                token_index = topi.item()
                token = code_dic_i2w[token_index]

                if token in method_list:  # replace
                    decoded_words.append(code_dic_i2w[topi.item()])
                    decoder_input = torch.tensor(node_onehot_t[K][token], dtype=torch.float32).to(device)
                else:  # Do not replace
                    decoded_words.append(code_dic_i2w[topi.item()])
                    decoder_input = topi.squeeze().detach()
                    # print(decoder_input)
                    # print(decoder_input.shape)

        return decoded_words, decoder_attentions[:di + 1]


def model_save(encoder1, embedder, attn_decoder1, ds_name):
    torch.save(encoder1.state_dict(), 'model/'+ds_name+'encoder_model.pkl')
    torch.save(embedder.state_dict(), 'model/'+ds_name+'embedder_model.pkl')
    torch.save(attn_decoder1.state_dict(), 'model/'+ds_name+'decoder_model.pkl')
