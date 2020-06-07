from __future__ import unicode_literals, print_function, division
import random
import torch
import torch.nn as nn
from torch import optim
import time
import math
import numpy as np
from eval_tools import Topk
from BeamSearchNode import BeamSearchNode
from queue import PriorityQueue
# from eval import MAX_LENGTH
import operator

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


def train_tge(input_tensor, target_tensor, encoder, embedder, decoder,
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


def trainTGE(encoder, embedder, decoder, n_iters,
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

        loss, node_onehot_t = train_tge(input_tensor, target_tensor, encoder, embedder, decoder,
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


def evaluate_tge(encoder, embedder, decoder,
                 sentence, max_length, node_onehot_t, code_dic_i2w, method_list, K, beam_search, beam_num):
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
        # decoder_attentions = torch.zeros(max_length, max_length)

        if not beam_search:
            for di in range(max_length):
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                # decoder_attentions[di] = decoder_attention.data

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
            return decoded_words

        else:  # beam search
            topk = 1  # how many sentence do you want to generate
            endnodes = []
            number_required = min((topk +1), topk-len(endnodes))
            node = BeamSearchNode(decoder_hidden, None, decoder_input, 0, 1)
            nodes = PriorityQueue()

            # start the queue
            nodes.put((-node.eval(), node))
            qsize = 1

            # decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            # # decoder_attention[0] = decoder_attention.data
            # topv, topi = decoder_output.data.topk(beam_num)  # i = index, SOS -> beam   dtype = tensor
            # decoder_input = topi[0]  # len = beam_num

            # start beam search
            while True:
                # give up when decoding takes too long
                if qsize > max_length: break

                # fetch the best node
                score, n = nodes.get()
                decoder_input = n.wordId
                decoder_hidden = n.h

                if n.wordId.item() == EOS_token and n.prevNode != None:
                    endnodes.append((score, n))
                    # if we reached maximum of sentences required
                    if len(endnodes) >= number_required:
                        break
                    else:
                        continue

                # decode for one step using decoder
                decoder_output, decoder_hidden, _ = decoder(decoder_input, decoder_hidden, encoder_outputs)

                # put here real beam search of top
                log_prob, indexes = torch.topk(decoder_output, beam_num)
                nextnodes = []

                for new_k in range(beam_num):
                    decoded_t = indexes[0][new_k].view(1, -1)
                    log_p = log_prob[0][new_k].item()

                    node = BeamSearchNode(decoder_hidden, n, decoded_t, n.logp + log_p, n.leng+1)
                    score = -node.eval()
                    nextnodes.append((score, node))

                # put them into queue
                for i in range(len(nextnodes)):
                    score, nn = nextnodes[i]
                    nodes.put((score, nn))
                    # increase qsize
                qsize += len(nextnodes) - 1

            # choose nbest paths, back trace them
            if len(endnodes) == 0:
                endnodes = [nodes.get() for _ in range(topk)]

            utterances = []
            for score, n in sorted(endnodes, key=operator.itemgetter(0)):
                utterance = []
                utterance.append(n.wordId)
                # back trace
                while n.prevNode != None:
                    n = n.prevNode
                    utterance.append(n.wordId)

                utterance = utterance[::-1]
                utterances.append(utterance)

            decoded_words = [code_dic_i2w[int(i[0][0])] for i in utterances[0]]
            return decoded_words

            # for di in range(1, max_length):  # each word
            #     all_candidates = []  # beam_num * len(dict), tempt
            #     for bn in range(beam_num):  # each beam
            #         decoder_output, decoder_hidden, _ = decoder(decoder_input[bn], decoder_hidden, encoder_outputs)
            #         v, i = decoder_output.data.topk(beam_num)
            #         for cc in range(beam_num):
            #             candidate = [int(i[0][cc]), float(v[0][cc])]
            #             all_candidates.append(candidate)
            #         # decoder_output_list.append(decoder_output)
            #     exit()
            #     ordered = sorted(all_candidates, key=lambda tup: tup[1])
            #     top_candi = ordered[:beam_num]  # 3
            #     print(top_candi)
            #     decoder_input = [torch.tensor(i[0]).to(device) for i in top_candi]  # to device
            #     print(decoder_input)
            #
            # exit()


def beam_decode(target_tensor, decoder_hiddens, decoder, encoder_outputs=None):
    '''
    :param target_tensor: target indexes tensor of shape [B, T] where B is the batch size and T is the maximum length of the output sentence
    :param decoder_hidden: input tensor of shape [1, B, H] for start of the decoding
    :param encoder_outputs: if you are using attention mechanism you can pass encoder outputs, [T, B, H] where T is the maximum length of input sentence
    :return: decoded_batch
    '''

    beam_width = 10
    topk = 1  # how many sentence do you want to generate
    decoded_batch = []

    # decoding goes sentence by sentence
    for idx in range(target_tensor.size(0)):
        if isinstance(decoder_hiddens, tuple):  # LSTM case
            decoder_hidden = (decoder_hiddens[0][:,idx, :].unsqueeze(0),decoder_hiddens[1][:,idx, :].unsqueeze(0))
        else:
            decoder_hidden = decoder_hiddens[:, idx, :].unsqueeze(0)
        encoder_output = encoder_outputs[:,idx, :].unsqueeze(1)

        # Start with the start of the sentence token
        decoder_input = torch.LongTensor([[SOS_token]], device=device)

        # Number of sentence to generate
        endnodes = []
        number_required = min((topk + 1), topk - len(endnodes))

        # starting node -  hidden vector, previous node, word id, logp, length
        node = BeamSearchNode(decoder_hidden, None, decoder_input, 0, 1)
        nodes = PriorityQueue()

        # start the queue
        nodes.put((-node.eval(), node))
        qsize = 1

        # start beam search
        while True:
            # give up when decoding takes too long
            if qsize > 2000: break

            # fetch the best node
            score, n = nodes.get()
            decoder_input = n.wordid
            decoder_hidden = n.h

            if n.wordid.item() == EOS_token and n.prevNode != None:
                endnodes.append((score, n))
                # if we reached maximum # of sentences required
                if len(endnodes) >= number_required:
                    break
                else:
                    continue

            # decode for one step using decoder
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_output)

            # PUT HERE REAL BEAM SEARCH OF TOP
            log_prob, indexes = torch.topk(decoder_output, beam_width)
            nextnodes = []

            for new_k in range(beam_width):
                decoded_t = indexes[0][new_k].view(1, -1)
                log_p = log_prob[0][new_k].item()

                node = BeamSearchNode(decoder_hidden, n, decoded_t, n.logp + log_p, n.leng + 1)
                score = -node.eval()
                nextnodes.append((score, node))

            # put them into queue
            for i in range(len(nextnodes)):
                score, nn = nextnodes[i]
                nodes.put((score, nn))
                # increase qsize
            qsize += len(nextnodes) - 1

        # choose nbest paths, back trace them
        if len(endnodes) == 0:
            endnodes = [nodes.get() for _ in range(topk)]

        utterances = []
        for score, n in sorted(endnodes, key=operator.itemgetter(0)):
            utterance = []
            utterance.append(n.wordid)
            # back trace
            while n.prevNode != None:
                n = n.prevNode
                utterance.append(n.wordid)

            utterance = utterance[::-1]
            utterances.append(utterance)

        decoded_batch.append(utterances)

    return decoded_batch


def greedy_decode(decoder, decoder_hidden, encoder_outputs, target_tensor, max_length):
    '''
    :param target_tensor: target indexes tensor of shape [B, T] where B is the batch size and T is the maximum length of the output sentence
    :param decoder_hidden: input tensor of shape [1, B, H] for start of the decoding
    :param encoder_outputs: if you are using attention mechanism you can pass encoder outputs, [T, B, H] where T is the maximum length of input sentence
    :return: decoded_batch
    '''

    batch_size, seq_len = target_tensor.size()
    decoded_batch = torch.zeros((batch_size, max_length))
    decoder_input = torch.LongTensor([[SOS_token] for _ in range(batch_size)], device=device)

    for t in range(max_length):
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)

        topv, topi = decoder_output.data.topk(1)  # get candidates
        topi = topi.view(-1)
        decoded_batch[:, t] = topi

        decoder_input = topi.detach().view(-1, 1)

    return decoded_batch


def model_save(encoder1, embedder, attn_decoder1, ds_name):
    torch.save(encoder1.state_dict(), 'model/'+ds_name+'encoder_model.pkl')
    torch.save(embedder.state_dict(), 'model/'+ds_name+'embedder_model.pkl')
    torch.save(attn_decoder1.state_dict(), 'model/'+ds_name+'decoder_model.pkl')
