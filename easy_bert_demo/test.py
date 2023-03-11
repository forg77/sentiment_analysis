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
df = pd.read_csv("usual_train.csv")
df1 = pd.read_csv("usual_eval_labeled.csv")
X = df['text'].values
y = df['label'].values
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, stratify=y, random_state=223)
X_test = np.array(df1['text'].values)
y_test = np.array(df1['label'].values)
ros = RandomOverSampler()  # 样本数量不均衡 采取超采样 duplicating some of the original samples of the minority class
X_train_os, y_train_os = ros.fit_resample(np.array(X_train).reshape(-1, 1), np.array(y_train).reshape(-1, 1))
X_train_os = X_train_os.flatten()
y_train_os = y_train_os.flatten()
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


train_inputs, train_masks = bert_tokenizer(X_train_os)
val_inputs, val_masks = bert_tokenizer(X_valid)
test_inputs, test_masks = bert_tokenizer(X_test)
train_labels = torch.from_numpy(y_train_os)
val_labels = torch.from_numpy(y_valid)
test_labels = torch.from_numpy(y_test)
# Create the DataLoader for our training set
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)

# Create the DataLoader for our validation set
val_data = TensorDataset(val_inputs, val_masks, val_labels)
val_sampler = SequentialSampler(val_data)
val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=BATCH_SIZE)

# Create the DataLoader for our test set
test_data = TensorDataset(test_inputs, test_masks, test_labels)
test_sampler = SequentialSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=BATCH_SIZE)


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

    # Set up optimizer
    optimizer = AdamW(bert_classifier.parameters(),
                      lr=5e-5,  # learning rate, set to default value
                      eps=1e-8  # decay, set to default value
                      )

    # ---- Set up learning rate scheduler ----#

    # Calculate total number of training steps
    total_steps = len(train_dataloader) * epochs

    # Define the scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,  # Default value
                                                num_training_steps=total_steps)
    return bert_classifier, optimizer, scheduler


bert_classifier, optimizer, scheduler = initialize_model(epochs=EPOCHS)
# Define Cross entropy Loss function for the multiclass classification task
bert_classifier.load_state_dict(torch.load('./model'))
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


bert_preds = bert_predict(bert_classifier, test_dataloader)
print('Classification Report for BERT :\n', classification_report(y_test, bert_preds, target_names=sentiments))
