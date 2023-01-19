import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel, BertForMaskedLM
from torchvision import datasets, models, transforms
from config import *
from transformers import AutoTokenizer, AutoModel

class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        self.res_model = models.resnet50(pretrained=True)
        self.new_model = torch.nn.Sequential(*list(self.res_model.children())[:-1])

    def forward(self, x):
        image_feature_map = self.new_model(x).data
        n = image_feature_map.shape[0]
        image_feature_map = image_feature_map.view(n, 2048, -1)
        image_vector = torch.mean(image_feature_map, -1)  # (n, 2048)
        image_feature_map = image_feature_map.permute(2, 0, 1)  # (49, n, 2048)
        return image_feature_map, image_vector


class model3_1(nn.Module):

    def __init__(self):
        super(model3_1, self).__init__()

        self.visual_model = ImageEncoder()  # 2048

        self.text_model = AutoModel.from_pretrained('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        # self.text_model = AutoModel.from_pretrained('../pretrained/sixtp_model')
        # self.text_model = AutoModel.from_pretrained('../pretrained/labse_bert_model')        
        for p in self.parameters():
            p.requires_grad = False

        self.img_vec = nn.Linear(2048, 200)
        self.text_vec = nn.Linear(768, 200)

        self.bn_input1 = nn.BatchNorm1d(384, momentum=0.5)
        self.bn_input2 = nn.BatchNorm1d(2048, momentum=0.5)

        self.ln_input = nn.LayerNorm(512, eps=1e-05, elementwise_affine=True)

        self.dropout = nn.Dropout(0.4)

        self.fc1 = nn.Linear(2048, 512)
        self.fc2 = nn.Linear(384, 512)

        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(1024, 512)
        self.classifier1 = nn.Linear(512, 13)
        self.classifier2 = nn.Linear(512, 13)

    def forward(self, text, img):
        text = self.text_model(text)[1]
        text = self.bn_input1(text)

        # text = self.dropout(text)
        img = self.visual_model(img)
        img = self.bn_input2(img[1])
        # img = img[1]
        # img = self.dropout(img)

        text = self.dropout(torch.tanh(self.fc2(text)))
        img = self.dropout(torch.tanh(self.fc1(img)))

        cat_vec = torch.cat([text,img], dim=1)

        cat_vec1 = torch.tanh(self.fc3(cat_vec))

        cat_vec1 = self.ln_input(cat_vec1)

        cat_vec2 = torch.tanh(self.fc4(cat_vec))

        cat_vec2 = self.ln_input(cat_vec2)

        out1 = F.softmax(self.classifier1(cat_vec1), dim=1)
        out2 = F.softmax(self.classifier2(cat_vec2), dim=1)

        return out1, out2