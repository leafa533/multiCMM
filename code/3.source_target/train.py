import pickle
import warnings
import time
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
import os, sys, json
import random
from models import *
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import trange
from sklearn.metrics import f1_score, accuracy_score, recall_score
from dataset import *
from transformers import AutoTokenizer, AutoModelForMaskedLM, BertTokenizer, BertModel
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

SEED=258
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic=True
warnings.filterwarnings('ignore')

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

def logging_func(logfile, message):
    with open(logfile, "a") as f:
        f.write(message)
    f.close()

class MultiClassFocalLossWithAlpha(nn.Module):
    def __init__(self, alpha, gamma=2, reduction='mean'):
        """
        :param alpha: 权重系数列表
        :param gamma: 困难样本挖掘的gamma
        :param reduction:
        """
        super(MultiClassFocalLossWithAlpha, self).__init__()
        self.device = torch.device("cuda:2")
        self.alpha = torch.tensor(alpha).to(self.device)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
        alpha = self.alpha[target]  # 为当前batch内的样本，逐个分配类别权重，shape=(bs), 一维向量
        log_softmax = torch.log_softmax(pred, dim=1) # 对模型裸输出做softmax再取log, shape=(bs, 3)
        logpt = torch.gather(log_softmax, dim=1, index=target.view(-1, 1))  # 取出每个样本在类别标签位置的log_softmax值, shape=(bs, 1)
        logpt = logpt.view(-1)  # 降维，shape=(bs)
        ce_loss = -logpt  # 对log_softmax再取负，就是交叉熵了
        pt = torch.exp(logpt)  #对log_softmax取exp，把log消了，就是每个样本在类别标签位置的softmax值了，shape=(bs)
        focal_loss = alpha * (1 - pt) ** self.gamma * ce_loss  # 根据公式计算focal loss，得到每个样本的loss值，shape=(bs)
        if self.reduction == "mean":
            return torch.mean(focal_loss)
        if self.reduction == "sum":
            return torch.sum(focal_loss)
        return focal_loss


def train(featype, language, mainSavedir, Resdir, best_model, ifgpu, iffinetune, datasetdir):
    Savedir = os.path.join(mainSavedir, featype)
    Resultsdir = os.path.join(Resdir, featype)
    if ifgpu == 'true':
        device = torch.device("cuda:2")
    else:
        device = 'cpu'
    if not os.path.exists(Savedir):
        os.makedirs(Savedir)

    if not os.path.exists(Resultsdir):
        os.makedirs(Resultsdir)

    ## Configuration
    TRAIN_BATCH_SIZE = 8
    EVAL_BATCH_SIZE = 8
    NUM_TRAIN_EPOCHS = 20
    NWORKER = 0
    lr = 3e-5
    finetunelr = 5e-6


    ##
    conf = {}
    conf.update({'adim': 768})
    conf.update({'middle_hidden_size': 512})
    conf.update({'bert_dropout': 0.4})
    conf.update({'resnet_dropout': 0.4})

    conf.update({'attention_dropout': 0.3})
    conf.update({'attention_nheads': 8})
    conf.update({'fuse_dropout': 0.5})
    conf.update({'out_hidden_size': 256})
    conf.update({'num_labels': 13})
    
    if(featype == 'bert'):
        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    elif(featype == 'xlm'):
        tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base') 

    train_data = MyDataSet(datasetdir, tokenizer, language, 'train')
    test_data = MyDataSet(datasetdir, tokenizer, language, 'test')
    length = len(train_data)
    print('train_example_length: ', length)


    model = Model(conf, device, featype)
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr, eps=1e-8, weight_decay=1e-5)

    data_loader_train = torch.utils.data.DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE,
                                                    num_workers=NWORKER,
                                                    shuffle=True,
                                                    drop_last=True,
                                                    collate_fn=pad_custom_sequence)
    data_loader_test = torch.utils.data.DataLoader(test_data, batch_size=EVAL_BATCH_SIZE,
                                                   num_workers=NWORKER,
                                                   collate_fn=pad_custom_sequence)
    # alpha1 = [0.9, 0.8, 0.55, 0.5, 0.5, 0.8, 0.8, 0.9, 0.5, 0.5, 0.55, 0.24, 0.8]
    # alpha2 = [0.9, 0.8, 0.5, 0.6, 0.6, 0.8, 0.8, 0.8, 0.5, 0.5, 0.6, 0.24, 0.8]
    # alpha1 = [0.8, 0.8, 0.5, 0.5, 0.5, 0.8, 0.8, 0.8, 0.5, 0.5, 0.5, 0.25, 0.8]
    # alpha1 = alpha2 = [0.8, 0.8, 0.6, 0.6, 0.6, 0.8, 0.8, 0.8, 0.6, 0.6, 0.6, 0.4, 0.8]
    alpha1 = [0.4, 0.4, 0.4, 0.4, 0.3, 0.4, 0.3, 0.4, 0.4, 0.2, 0.3, 0.1, 0.4]
    alpha2 = [0.4, 0.4, 0.3, 0.4, 0.3, 0.4, 0.4, 0.4, 0.4, 0.2, 0.3, 0.1, 0.4]

    criterion1 = MultiClassFocalLossWithAlpha(alpha=alpha1)
    criterion2 = MultiClassFocalLossWithAlpha(alpha=alpha2)
    # criterion = nn.CrossEntropyLoss()
    
    print('alpha1: ', alpha1)
    print('alpha2: ', alpha2)
    for epoch in trange(int(NUM_TRAIN_EPOCHS), desc='Epoch'):
        since = time.time()
        train_loss = 0
        model.train()
        train_label1 = []
        train_predict1 = []
        train_label2 = []
        train_predict2 = []
        for i, data in enumerate(data_loader_train):
            indexs, texts, texts_mask, images, labels1, labels2 = data
            # indexs, texts, texts_mask, labels = data

            texts = texts.to(device)
            texts_mask = texts_mask.to(device)
            images = images.to(device)
            labels1 = labels1.to(device)
            labels2 = labels2.to(device)
            out1, out2 = model(texts, texts_mask, images)


            loss1 = criterion1(out1, labels1)
            loss2 = criterion2(out2, labels2)
            loss = (loss1 + loss2)/2
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #标签一
            train_label1.extend(labels1.cpu().data.tolist())
            train_predict1.extend(torch.argmax(out1, dim=1).cpu().detach().tolist())
            # train_predict1.extend(out1.cpu().detach().tolist())
            #标签二
            train_label2.extend(labels2.cpu().data.tolist())
            train_predict2.extend(torch.argmax(out2, dim=1).cpu().detach().tolist())
        train_acc1 = accuracy_score(train_label1, train_predict1)
        train_acc2 = accuracy_score(train_label2, train_predict2)
        torch.cuda.empty_cache()
        time_elapsed = time.time() - since
        print('epoch {} complete in {:.0f}m {:.0f}s'.format(epoch, time_elapsed // 60, time_elapsed % 60))
        print('train_loss:'+str(train_loss))
        print('train_acc1: {}    train_acc2: {}'.format(train_acc1, train_acc2))
        # if epoch%2 == 0 or epoch == int(NUM_TRAIN_EPOCHS)-1:
        if epoch != 100:
            print('*'*10 + 'Test' + '*'*10)
            acc1, acc2, reca1, reca2, f1_1, f1_2, loss = eval(model, device, criterion1, criterion2, data_loader_test, Resultsdir, epoch)
            message = "Epoch %d:   target: (%f,%f,%f)   source: (%f,%f,%f)  loss: %f \n" % (epoch, acc1, reca1, f1_1, acc2, reca2, f1_2, loss)
            log_file = os.path.join(Resultsdir, "performance.log")
            logging_func(log_file, message)
            print('Test acc1:{} reca1:{} f1score1:{} loss:{}'.format(acc1, reca1, f1_1, loss.item()))
            print('Test acc2:{} reca2:{} f1score2:{}'.format(acc2, reca2, f1_2))


def eval(model, device, criterion1, criterion2, loader, resdir, epoch):
    model.eval()
    loss_total = 0
    predict1_all = []
    labels1_all = []
    predict2_all = []
    labels2_all = []
    with torch.no_grad():
        for i, data in enumerate(loader):
            # 将数据从 train_loader 中读出来,一次读取的样本数是batch_size个
            indexs, texts, texts_mask, images, labels1, labels2 = data
            # indexs, texts, texts_mask, labels = data
            texts = texts.to(device)
            texts_mask = texts_mask.to(device)
            images = images.to(device)
            labels1 = labels1.to(device)
            labels2 = labels2.to(device)
            out1, out2 = model(texts, texts_mask, images)
            # out = model(texts, texts_mask)

            loss1 = criterion1(out1, labels1)
            loss2 = criterion2(out2, labels2)
            loss_total += (loss1 + loss2)/2

            labels1_all.extend(labels1.cpu().data.tolist())
            predict1_all.extend(torch.argmax(out2, dim=1).cpu().detach().tolist())
            labels2_all.extend(labels2.cpu().data.tolist())
            predict2_all.extend(torch.argmax(out2, dim=1).cpu().detach().tolist())
        acc1 = accuracy_score(labels1_all, predict1_all)
        acc2 = accuracy_score(labels2_all, predict2_all)
        f1_1 = f1_score(labels1_all, predict1_all, average='macro')
        f1_2 = f1_score(labels2_all, predict2_all, average='macro')
        recall1 = recall_score(labels1_all, predict1_all, average='macro')
        recall2 = recall_score(labels2_all, predict2_all, average='macro')
        np.savetxt(os.path.join(resdir, 'predict_target_'+ str(epoch) + '.txt'), predict1_all, fmt='%d', newline=" ")
        np.savetxt(os.path.join(resdir, 'origin_target' + '.txt'), labels1_all, fmt='%d', newline=" ")
        np.savetxt(os.path.join(resdir, 'predict_source_'+ str(epoch) + '.txt'), predict2_all, fmt='%d', newline=" ")
        np.savetxt(os.path.join(resdir, 'origin_source' + '.txt'), labels2_all, fmt='%d', newline=" ")
        return acc1, acc2, recall1, recall2, f1_1, f1_2, loss_total


if __name__ == '__main__':
    train(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7], sys.argv[8])
