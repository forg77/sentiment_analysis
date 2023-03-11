import pandas as pd
import numpy as np
import time
import torch
from sklearn.metrics import classification_report
from torch import nn
from torch.utils.data import TensorDataset, RandomSampler, DataLoader, SequentialSampler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from transformers import BertModel
from transformers import BertTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup

# usual_train
# 0 neutral 5749
# 1 angry 8344
# 2 happy 5379
# 3 surprise 2086
# 4 sad 4990
# 5 fear 1220
sentiments = ["neutral", "angry", "happy", "surprise", "sad", "fear"]

MAX_LEN = 256  # 处理的最长长度
BATCH_SIZE = 40
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCHS = 4
Learning_Rate = 5e-5
FALSE_IDS = np.zeros(256, )

tokenizer = BertTokenizer.from_pretrained('hfl/chinese-macbert-base')


def bert_tokenizer(data):
    input_ids = []
    attention_masks = []
    for sent in data:
        try:
            encoded_sent = tokenizer.encode_plus(
                text=sent,
                add_special_tokens=True,  # Add `[CLS]` and `[SEP]` special tokens
                max_length=MAX_LEN,
                pad_to_max_length=True,  # Pad sentence to max length
                return_attention_mask=True  # Return attention mask
            )
            # k1 = encoded_sent.get('input_ids')
            # print(k1)
            # [101, 976, 749, 702, 679, 1962, 4638, 3457, 8024, 4994, 4197, 3193, 677, 6629, 3341, 2970, 4708, 2418, 7741, 749, 8024, 2571, 2828, 2769, 3698, 4156, 749, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            input_ids.append(encoded_sent.get('input_ids'))
            # k2 = encoded_sent.get('attention_mask')
            # print(k2)
            # [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            attention_masks.append(encoded_sent.get('attention_mask'))
        except Exception as e:
            print(e)
            input_ids.append(FALSE_IDS)
            attention_masks.append(FALSE_IDS)
            continue

    # Convert lists to tensors
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)
    # print('###',input_ids.shape,'###')
    ### torch.Size([40050, 256]) ###
    ### torch.Size([5554, 256]) ###
    ### torch.Size([2000, 256]) ###
    # print('###',attention_masks.shape,'###')
    ### torch.Size([40050, 256]) ###
    ### torch.Size([5554, 256]) ###
    ### torch.Size([2000, 256]) ###
    return input_ids, attention_masks






class BertClassifier(nn.Module):
    def __init__(self, freeze_bert=False):
        super(BertClassifier, self).__init__()
        # Specify hidden size of BERT, hidden size of the classifier, and number of labels
        n_input = 768
        n_hidden = 100
        n_output = 6

        # Instantiate BERT model
        self.bert = BertModel.from_pretrained('hfl/chinese-macbert-base')

        # Add dense layers to perform the classification
        self.classifier = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_output)
        )
        # Add possibility to freeze the BERT model
        # to avoid fine tuning BERT params (usually leads to worse results)
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        # Feed input data to BERT
        input_ids = input_ids.to(torch.int64)
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)
        # print(outputs,'11111')
        # Extract the last hidden state of the token `[CLS]` for classification task
        last_hidden_state_cls = outputs[0][:, 0, :]
        # print(last_hidden_state_cls.shape)
        # Feed input to classifier to compute logits
        logits = self.classifier(last_hidden_state_cls)

        return logits


def initialize_model(epochs=4):
    # Instantiate Bert Classifier
    bert_classifier = BertClassifier(freeze_bert=False)

    bert_classifier.to(DEVICE)
    bert_classifier.load_state_dict(torch.load('.\model'))
    return bert_classifier


bert_classifier = initialize_model()
# Define Cross entropy Loss function for the multiclass classification task

loss_fn = nn.CrossEntropyLoss()


def bert_predict(model, test_dataloader):
    # Define empty list to host the predictions
    preds_list = []

    # Put the model into evaluation mode
    model.eval()

    for batch in test_dataloader:
        batch_input_ids, batch_attention_mask = tuple(t.to(DEVICE) for t in batch)[:2]

        # Avoid gradient calculation of tensors by using "no_grad()" method
        with torch.no_grad():
            logit = model(batch_input_ids, batch_attention_mask)
            # print(logit.shape) 40个 6种
        # Get index of highest logit
        pred = torch.argmax(logit, dim=1).cpu().numpy()
        # Append predicted class to list
        preds_list.extend(pred)

    return preds_list

def work(listd,listc,model=bert_classifier):
    d_inputs, d_masks = bert_tokenizer(listd)
    lend = len(listd)
    cur = 0
    d_preds = []
    while cur < lend:
        # print(cur)
        end = min(cur+BATCH_SIZE,lend)
        d_input = d_inputs[cur:end]
        d_input = d_input.to(DEVICE)
        d_mask = d_masks[cur:end]
        d_mask = d_mask.to(DEVICE)
        cur += BATCH_SIZE
        with torch.no_grad():
            logit = model(d_input, d_mask)
        d_pred = torch.argmax(logit,dim=1).cpu().numpy()
        d_preds.extend(d_pred)
    c_inputs, c_masks = bert_tokenizer(listc)
    lenc = len(listc)
    cur = 0
    c_preds = []
    while cur < lenc:
        end = min(cur + BATCH_SIZE, lenc)
        c_input = c_inputs[cur:end]
        c_input = c_input.to(DEVICE)
        c_mask = c_masks[cur:end]
        c_mask = c_mask.to(DEVICE)
        cur += BATCH_SIZE
        with torch.no_grad():
            logit = model(c_input, c_mask)
        c_pred = torch.argmax(logit, dim=1).cpu().numpy()
        c_preds.extend(c_pred)

    return d_preds,c_preds

