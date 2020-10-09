from run import *
from model import *
import nltk

print('evaluation begin...')
encoder = EncoderRNN(len(describe_dic_i2w), hidden_size).to(device)
embedder = EmbedderRNN(len(code_dic_i2w), len(code_dic_i2w), dropout=0.1).to(device)
attn_decoder = AttnDecoderRNN(hidden_size, len(code_dic_i2w), dropout_p=0.1,
                              max_length=max(max_source_len, max_target_len)).to(device)

encoder.load_state_dict(torch.load('model/' + ds_name + 'encoder_model.pkl'))
embedder.load_state_dict(torch.load('model/' + ds_name + 'embedder_model.pkl'))
attn_decoder.load_state_dict(torch.load('model/' + ds_name + 'decoder_model.pkl'))

for i in range(num_test):
    input_tensor = test_source[i]
    ground_truth_tenor = test_target[i]
    input_words = index2text(input_tensor, describe_dic_i2w)
    ground_truth_words = index2text(ground_truth_tenor, code_dic_i2w)
    output_words = evaluate_tge(encoder, embedder, attn_decoder, input_tensor,
                                max_length=MAX_LENGTH,
                                node_onehot_t=node_onehot_t, code_dic_i2w=code_dic_i2w,
                                method_list=method_list, K=K,
                                beam_search=is_beamsearch, beam_num=beam_num)
    bleu = nltk.translate.bleu_score.sentence_bleu([str(ground_truth_words).split(' ')], str(output_words).split(' '))

    print(" input: %s \n output: %s\n ground truth: %s\n BLEU: %.8f \n"
          % (' '.join(input_words), ' '.join(output_words), ' '.join(ground_truth_words), bleu))
