CONFIG = {'CONFIG_NAME': 'glu-gan2',

'DATASET_NAME': 'coco',
'DATA_DIR': '../coco/',
'GPU_ID': 0,
'WORKERS': 8,
'CUDA' : True,
'RNN_TYPE' : 'LSTM',
'B_VALIDATION' : False,
'DEVICE' : 'cuda:0',

'TREE':{
    'BRANCH_NUM': 3,
    'BASE_SIZE' : 64},


'TRAIN':{
    'FLAG': False,
    'NET_G': '../Model/netG_epoch_26.pth',  # '../models/coco_AttnGAN2.pth'
    'B_NET_D': True,
    'BATCH_SIZE': 12,  # 32
    'MAX_EPOCH': 100,
    'SNAPSHOT_INTERVAL': 1,
    'DISCRIMINATOR_LR': 0.00001,
    'GENERATOR_LR': 0.0005,
    'ENCODER_LR' : 0.0002,
    'RNN_GRAD_CLIP' : 0.25,
    'B_NET_D' : True,
    #   p
    'NET_E': '../DAMSMencoders/text_encoder63.pth',#'../DAMSMencoders/coco/text_encoder100.pth'
    'SMOOTH':
    {
        'GAMMA1': 4.0,  # 1,2,5 good 4 best  10&100bad
        'GAMMA2': 5.0,
        'GAMMA3': 10.0,  # 10good 1&100bads
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