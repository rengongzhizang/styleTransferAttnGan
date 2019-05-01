from __future__ import print_function


from datasets import TextDataset
from trainer import condGANTrainer as trainer
from cfg.config import CONFIG
import re
import os
import sys
import time
import random
import pprint
import datetime
import dateutil.tz
import argparse
import numpy as np
import json
import torch
import torchvision.transforms as transforms

dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)


def gen_example(wordtoix, algo):
    '''generate images from example sentences'''
    from nltk.tokenize import RegexpTokenizer
    filepath = '../Model/example_filenames.txt'
    data_dic = {}
    path = '../coco/' + 'captions_train2014.json'
    caption_dict = json.load(open(path,'r'))
    cd = {}
    
    for im in caption_dict['annotations']:
        cd[im['image_id']] = im['caption']
    with open(filepath, "r") as f:
        for name in f:
            if len(name) == 0:
                continue
            filepath = '../coco/train/' + name.strip()
            with open(filepath, "r") as f:
                print('Load from:', name)
                im_id = int(re.sub("[^0-9]", "", name)[-6:])
                sentences = cd[im_id].split('\n')
                print(sentences)
                # a list of indices for a sentence
                captions = []
                cap_lens = []
                for sent in sentences:
                    if len(sent) == 0:
                        continue
                    sent = sent.replace("\ufffd\ufffd", " ")
                    tokenizer = RegexpTokenizer(r'\w+')
                    tokens = tokenizer.tokenize(sent.lower())
                    if len(tokens) == 0:
                        print('sent', sent)
                        continue

                    rev = []
                    for t in tokens:
                        t = t.encode('ascii', 'ignore').decode('ascii')
                        if len(t) > 0 and t in wordtoix:
                            rev.append(wordtoix[t])
                    captions.append(rev)
                    cap_lens.append(len(rev))
            max_len = np.max(cap_lens)

            sorted_indices = np.argsort(cap_lens)[::-1]
            cap_lens = np.asarray(cap_lens)
            cap_lens = cap_lens[sorted_indices]
            cap_array = np.zeros((len(captions), max_len), dtype='int64')
            for i in range(len(captions)):
                idx = sorted_indices[i]
                cap = captions[idx]
                c_len = len(cap)
                cap_array[i, :c_len] = cap
            key = name[(name.rfind('/') + 1):]
            data_dic[key] = [cap_array, cap_lens, sorted_indices]
            
    algo.gen_example(data_dic)

def main():
    start_t = time.time()
    seed = 12345
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if CONFIG['CUDA']:
        torch.cuda.manual_seed_all(seed)
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')

    output_dir = '../output/%s_%s_%s' % \
    (CONFIG['DATASET_NAME'], CONFIG['CONFIG_NAME'], timestamp)

    split_dir, bshuffle = 'train', True
    if not CONFIG['TRAIN']['FLAG']:
        # bshuffle = False
        split_dir = 'test'

    imsize = CONFIG['TREE']['BASE_SIZE'] * (2 ** (CONFIG['TREE']['BRANCH_NUM'] - 1))
    image_transform = transforms.Compose([
        transforms.Scale(int(imsize * 76 / 64)),
        transforms.RandomCrop(imsize),
        transforms.RandomHorizontalFlip()])

    dataset = TextDataset(CONFIG['DATA_DIR'], split_dir,
                        base_size=CONFIG['TREE']['BASE_SIZE'],
                        transform=image_transform)
    
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=CONFIG['TRAIN']['BATCH_SIZE'],
        drop_last=True, shuffle=bshuffle, num_workers=int(CONFIG['WORKERS']))
    algo = trainer(output_dir, dataloader, dataset.n_words, dataset.ixtoword)
    if CONFIG['TRAIN']['FLAG']:
        algo.train()
    else:
        '''generate images from pre-extracted embeddings'''
        if CONFIG['B_VALIDATION']:
            algo.sampling(split_dir)  # generate images for the whole valid dataset
        else:
            gen_example(dataset.wordtoix, algo)  # generate images for customized /
            
        
    end_t = time.time()
    print('Total time for training:', end_t - start_t)


if __name__ == "__main__":
    main()