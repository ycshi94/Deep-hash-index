
def config_JointEmbeder():   
    conf = {
        # data_params
        'dataset_name':'CodeSearchDataset',
		'code_vec':'code_train.h5',
        'desc_vec': 'desc_train.h5',
        'code_vec_valid': 'code_valid.h5',
        'desc_vec_valid': 'desc_valid.h5',
        'code_vec_test': 'code_test.h5',
        'desc_vec_test': 'desc_test.h5',
        'code_len': 512,
        'desc_len': 512,


			   
        #parameters
        'name_len': 6,
        'tokens_len':50,
        'graphseq_len': 80,
        'n_words': 10000,

        #training_params            
        'batch_size':64,
        'chunk_size':200000,
        'nb_epoch': 2001,
		
        #'optimizer': 'adam',
        'learning_rate':2.08e-4,
        'adam_epsilon':1e-8,
        'warmup_steps':5000,
        'fp16': False,
        'fp16_opt_level': 'O1', 

        # model_params
        'emb_size': 512,
        'hash_len': 128,
        'n_hidden': 512,
        'lstm_dims': 256,      
        'margin': 0.3986,
        'sim_measure':'hash',
    }
    return conf



