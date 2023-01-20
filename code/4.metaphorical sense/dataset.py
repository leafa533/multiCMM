import csv
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from PIL import Image

from torchvision import transforms
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def pad_custom_sequence(sequence):
    id_sequence = [b[0] for b in sequence]
    text_sequence = [torch.LongTensor(b[1]) for b in sequence]
    # label_sequence = torch.LongTensor([b[3] for b in sequence], dtype=torch.int64)
    image_sequence = torch.FloatTensor([np.array(b[2]).tolist() for b in sequence])
    label_sequence = torch.LongTensor([b[3] for b in sequence])


    """处理文本， 统一长度"""
    text_mask = [torch.ones_like(text) for text in text_sequence]
    paded_texts = pad_sequence(text_sequence, batch_first=True, padding_value=0)
    paded_mask_sequence = pad_sequence(text_mask, batch_first=True, padding_value=0).gt(0)

    return id_sequence, paded_texts, paded_mask_sequence, image_sequence, label_sequence


class MyDataSet(Dataset):

    def __init__(self, path, tokenizer, language, dset):
        """可以在初始化函数当中对数据进行一些操作，比如读取、归一化等"""
        # 数据集初始化  根据id获取文本 图片 图片实体 文本关键词 源域 目标域
        self.tokenizer = tokenizer
        # self.language = language

        index = []
        texts_en = []
        imgs = []
        labels = []



        if dset == 'train':
            file = 'ground_train.csv'
        elif dset == 'test':
            file = 'ground_test.csv'
        with open(os.path.join('../data/csv', file), 'r', encoding='utf_8_sig') as f:
            f.readline()  # 跳过首行
            f_csv = csv.reader(f)
            for row in f_csv:
                index.append(row[0])
                texts_en.append(row[2])
                labels.append(row[9])
                imgs.append('../data/images/' + str(int(float(row[1]))) + '.jpg')
        self.index = index
        self.text_en = texts_en
        self.labels = labels
        self.imgs = imgs

    def __len__(self):
        """返回数据集当中的样本个数"""
        return len(self.text_en)

    def __getitem__(self, index):
        """返回样本集中的第 index 个样本；输入变量在前，输出变量在后"""
        text = self.text_en[index]


        tokens = self.tokenizer(text, truncation='longest_first')
        # tokens = self.tokenizer.tokenize('[CLS]' + text + '[SEP]')
        # encoded_text = self.tokenizer.convert_tokens_to_ids(tokens)
        encoded_text = tokens.data['input_ids']
        img = transforms.Compose([
            transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
            transforms.RandomRotation(degrees=15),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])(Image.open(self.imgs[index]).convert('RGB'))
        label = int(float(self.labels[index]))


        return index, encoded_text, img, label

 
