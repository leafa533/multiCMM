import torch
import torch.nn.functional as F
from transformers import XLMRobertaModel, BertModel, logging
from torchvision.models import resnet50
logging.set_verbosity_warning()


class BERTC(torch.nn.Module):
    def __init__(self, conf):
        torch.nn.Module.__init__(self)
        self.BERTtext = BertModel.from_pretrained('bert-base-multilingual-cased')
        self.textlinear = torch.nn.Sequential(
            torch.nn.Linear(conf['adim'], conf['middle_hidden_size']),
            # torch.nn.BatchNorm1d(conf['middle_hidden_size']),
            torch.nn.Dropout(conf['bert_dropout']),
            torch.nn.ReLU(inplace=True)
        )

    def forward(self, texts, masks):

        outputs = self.BERTtext(texts, masks)
        clsfeat = outputs[1]
        #clsfeat: torch.Size([batch_size, 768])

        x = self.textlinear(clsfeat)
        return x

class XLM(torch.nn.Module):
    def __init__(self, conf):
        torch.nn.Module.__init__(self)
        self.BERTtext = XLMRobertaModel.from_pretrained('xlm-roberta-base')
        self.textlinear = torch.nn.Sequential(
            torch.nn.Dropout(conf['bert_dropout']),
            torch.nn.Linear(conf['adim'], conf['middle_hidden_size']),
            torch.nn.ReLU(inplace=True)
        )

    def forward(self, texts, masks):
        outputs = self.BERTtext(texts, masks, return_dict=False)
        clsfeat = outputs[1]
        x = self.textlinear(clsfeat)    
        return x


class ResnetImage(torch.nn.Module):
    def __init__(self, conf):
        torch.nn.Module.__init__(self)
        self.full_resnet = resnet50(pretrained=True)
        self.resnet = torch.nn.Sequential(
            *(list(self.full_resnet.children())[:-1]),
            torch.nn.Flatten()
        )
        self.imgencoder = torch.nn.Sequential(
            torch.nn.Linear(self.full_resnet.fc.in_features, conf['middle_hidden_size'] * 2),
            torch.nn.BatchNorm1d(conf['middle_hidden_size'] * 2),
            torch.nn.Dropout(conf['resnet_dropout']),
            torch.nn.Linear(conf['middle_hidden_size'] * 2, conf['middle_hidden_size']),
            torch.nn.ReLU(inplace=True)
        )

    def forward(self, image):

        imagefeat = self.resnet(image)

        x = self.imgencoder(imagefeat)
        return x

class Model(torch.nn.Module):
    def __init__(self, conf, device, textmodel):
        torch.nn.Module.__init__(self)
        self.device = device
        if textmodel == 'bert':
            self.textmodel = BERTC(conf)
        elif textmodel == 'xlm':
            self.textmodel = XLM(conf)
        self.imgmodel = ResnetImage(conf)

        # attention
        self.attention = torch.nn.TransformerEncoderLayer(
            d_model=conf['middle_hidden_size'] * 2,
            nhead=conf['attention_nheads'],
            dropout=conf['attention_dropout']
        )

        #FC
        self.classifier = torch.nn.Sequential(
            # torch.nn.Dropout(conf['fuse_dropout']),
            torch.nn.Linear(conf['middle_hidden_size'] * 2, conf['out_hidden_size']),
            # torch.nn.Linear(conf['middle_hidden_size'], conf['out_hidden_size']),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(conf['fuse_dropout']),
            torch.nn.Linear(conf['out_hidden_size'], conf['num_labels'])
        )



    # def forward(self, texts, texts_mask, imgs, labels=None):
    def forward(self, texts, texts_mask, imgs, labels=None):

        text_feature = self.textmodel(texts, texts_mask)
        img_feature = self.imgmodel(imgs)

        attention_out = self.attention(torch.cat([text_feature, img_feature], dim=1))
        prob_vec = self.classifier(attention_out)
        

        return prob_vec


