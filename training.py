from run import *


print('\n\ntraining begin...')
node_onehot_t = trainDFG(encoder, embedder, attn_decoder, num_training,
                         training_inputs=training_source, training_outputs=training_target,
                         method_list=method_list, node_list_onehot_dict=node_list_onehot_dict, K=K,
                         behind_call_dict=behind_call_dict, front_call_dict=front_call_dict,
                         node_onehot_t=node_onehot_t, code_dic_i2w=code_dic_i2w,
                         print_every=20, learning_rate=0.01, max_length=max(max_source_len, max_target_len))

model_save(encoder, embedder, attn_decoder, ds_name)
print('models have been saved')
