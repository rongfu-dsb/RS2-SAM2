import os
import sys
import torch.utils.data as data
import torch
from torchvision import transforms
from torch.autograd import Variable
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
import random

# from bert.tokenization_bert import BertTokenizer

import h5py
from refer.refer import REFER

from args import get_parser
import pickle
import cv2

# Dataset configuration initialization
parser = get_parser()
args = parser.parse_args()

data_root = "/horizon-bucket/saturn_v_dev/01_users/fu.rong/RefSegRS"

def build_rsris_batches(setname):
    im_dir1 = f'{data_root}/images/'
    seg_label_dir = f'{data_root}/masks/'
    if setname == 'train':
        setfile = 'output_phrase_train.txt'
    if setname == 'val':
        setfile = 'output_phrase_val.txt'
    if setname == 'test':
        setfile = 'output_phrase_test.txt'

    n_batch = 0
    train_ids = []
    tf = f'{data_root}/'+setfile
    nn = 0
    imgnames = set()
    imname = 'start'
    all_imgs1 = []
    all_labels = []
    all_sentences = []

    test_sentence = []

    with open(tf,'r') as rf:
        rlines = rf.readlines()
        for idx,line in enumerate(rlines):
            lsplit = line.split(' ')
            if True:
                im_name1 = im_dir1 + lsplit[0] + '.tif'
                seg = seg_label_dir + lsplit[0] + '.tif'
                del(lsplit[0])
                if False and setname != 'train':
                    del(lsplit[-1])
                sentence = ' '.join(lsplit)
                sent = sentence

                im_1 = im_name1
                label_mask = seg
                all_imgs1.append(im_name1)
                all_labels.append(label_mask)
                all_sentences.append(sent)

    print("Dataset Loaded.")
    return all_imgs1, all_labels, all_sentences



class ReferDataset(data.Dataset):

    def __init__(self,
                 args,
                 image_transforms=None,
                 image_transforms_evf=None,
                 target_transforms=None,
                 split='train',
                 eval_mode=False):

        self.classes = []
        self.image_transforms = image_transforms
        self.image_transforms_evf = image_transforms_evf
        self.target_transform = target_transforms
        self.split = split
        self.max_tokens = 20

        all_imgs1, all_labels, all_sentences = build_rsris_batches(self.split)
        self.sentences = all_sentences
        self.imgs1 = all_imgs1
        self.labels = all_labels

        # self.input_ids = []
        # self.attention_masks = []
        # self.tokenizer = BertTokenizer.from_pretrained(args.bert_tokenizer)
        self.caption = []
        self.eval_mode = eval_mode
        # if we are testing on a dataset, test all sentences of an object;
        # o/w, we are validating during training, randomly sample one sentence for efficiency
        for r in range(len(self.imgs1)):
            img_sentences = [self.sentences[r]]
            # sentences_for_ref = []
            # attentions_for_ref = []
            caption_for_ref = []
            for i, el in enumerate(img_sentences):
                sentence_raw = el
                # attention_mask = [0] * self.max_tokens
                # padded_input_ids = [0] * self.max_tokens

                # input_ids = self.tokenizer.encode(text=sentence_raw, add_special_tokens=True)

                # truncation of tokens
                # input_ids = input_ids[:self.max_tokens]

                # padded_input_ids[:len(input_ids)] = input_ids
                # attention_mask[:len(input_ids)] = [1]*len(input_ids)

                # sentences_for_ref.append(torch.tensor(padded_input_ids).unsqueeze(0))
                # attentions_for_ref.append(torch.tensor(attention_mask).unsqueeze(0))
                caption_for_ref.append(sentence_raw)
            self.caption.append(caption_for_ref)
            # self.input_ids.append(sentences_for_ref)
            # self.attention_masks.append(attentions_for_ref)

    def get_classes(self):
        return self.classes

    def __len__(self):
        return len(self.imgs1)

    def __getitem__(self, index):
        this_img1 = self.imgs1[index]

        img = Image.open(this_img1).convert("RGB")
        #img1 = cv2.imread(this_img1)
        label_mask = cv2.imread(self.labels[index],2)
        #label_mask = Image.open(self.labels[index]).convert('L')

        ref_mask = np.array(label_mask) > 50
        annot = np.zeros(ref_mask.shape)
        annot[ref_mask == 1] = 1

        annot = Image.fromarray(annot.astype(np.uint8), mode="P")

        if self.image_transforms is not None:
            # resize, from PIL to tensor, and mean and std normalization
            img_evf = img
            img, target = self.image_transforms(img, annot)
            img_evf, _ = self.image_transforms_evf(img_evf, annot)

        if self.eval_mode:
            cap = []
            for s in range(len(self.caption[index])):
                c = self.caption[index][s]
                cap.append(c)
            # embedding = []
            # att = []
            # for s in range(len(self.input_ids[index])):
            #     e = self.input_ids[index][s]
            #     a = self.attention_masks[index][s]
            #     embedding.append(e.unsqueeze(-1))
            #     att.append(a.unsqueeze(-1))
            #
            # tensor_embeddings = torch.cat(embedding, dim=-1)
            # attention_mask = torch.cat(att, dim=-1)
        else:
            choice_sent = np.random.choice(len(self.caption[index]))
            cap = self.caption[index][choice_sent]
            # choice_sent = np.random.choice(len(self.input_ids[index]))
            # tensor_embeddings = self.input_ids[index][choice_sent]
            # attention_mask = self.attention_masks[index][choice_sent]

        return img, img_evf, target, cap
