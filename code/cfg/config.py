CONFIG = {'CONFIG_NAME': 'glu-gan2',

'DATASET_NAME': 'coco',
'DATA_DIR': '../coco/',
'GPU_ID': 0,
'WORKERS': 4,
'CUDA' : True,
'RNN_TYPE' : 'LSTM',
'B_VALIDATION' : False,

'TREE':{
    'BRANCH_NUM': 3,
    'BASE_SIZE' : 64},


'TRAIN':{
    'FLAG': True,
    'NET_G': '',  # '../models/coco_AttnGAN2.pth'
    'B_NET_D': True,
    'BATCH_SIZE': 14,  # 32
    'MAX_EPOCH': 120,
    'SNAPSHOT_INTERVAL': 5,
    'DISCRIMINATOR_LR': 0.0002,
    'GENERATOR_LR': 0.0002,
    'ENCODER_LR' : 0.0002,
    'RNN_GRAD_CLIP' : 0.25,
    'B_NET_D' : True,
    #
    'NET_E': '../DAMSMencoders/text_encoder5.pth',#'../DAMSMencoders/coco/text_encoder100.pth'
    'SMOOTH':
    {
        'GAMMA1': 4.0,  # 1,2,5 good 4 best  10&100bad
        'GAMMA2': 5.0,
        'GAMMA3': 10.0,  # 10good 1&100bad
        'LAMBDA': 50.0
    }
},

'GAN':{
    'DF_DIM': 96,
    'GF_DIM': 48,
    'SF_DIM': 1,
    'Z_DIM': 100,
    'R_NUM': 3,
    'CONDITION_DIM' : 100,
    'B_ATTENTION' : True,
    'B_DCGAN' : False
},

'TEXT':{
    'EMBEDDING_DIM': 256,
    'CAPTIONS_PER_IMAGE': 5,
    'WORDS_NUM': 12
}
}