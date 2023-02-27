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
FALSE_IDS = np.zeros(256,)
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
            input_ids.append(encoded_sent.get('input_ids'))
            attention_masks.append(encoded_sent.get('attention_mask'))
        except Exception as e:
            print(e)
            input_ids.append(FALSE_IDS)
            attention_masks.append(FALSE_IDS)
            continue

    # Convert lists to tensors
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)

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

        # Extract the last hidden state of the token `[CLS]` for classification task
        last_hidden_state_cls = outputs[0][:, 0, :]
        #print(last_hidden_state_cls.shape)
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
loss_fn = nn.CrossEntropyLoss()


def BertTrain(model, train_dataloader, val_dataloader, epochs, evaluation=True):
    for epoch_i in range(epochs):
        print("-" * 10)
        print("Epoch : {}".format(epoch_i + 1))
        print("-" * 10)
        print("-" * 38)
        print(f"{'BATCH NO.':^7} | {'TRAIN LOSS':^12} | {'ELAPSED (s)':^9}")
        print("-" * 38)
        # Measure the elapsed time of each epoch
        t0_epoch, t0_batch = time.time(), time.time()

        # Reset tracking variables at the beginning of each epoch
        total_loss, batch_loss, batch_counts = 0, 0, 0

        ###TRAINING###

        # Put the model into the training mode
        model.train()
        for step, batch in enumerate(train_dataloader):
            batch_counts += 1

            b_input_ids, b_attn_mask, b_labels = tuple(t.to(DEVICE) for t in batch)
            if b_input_ids == FALSE_IDS:
                continue
            # Zero out any previously calculated gradients
            model.zero_grad()

            # Perform a forward pass and get logits.
            logits = model(b_input_ids, b_attn_mask)

            # Compute loss and accumulate the loss values
            loss = loss_fn(logits, b_labels)
            batch_loss += loss.item()
            total_loss += loss.item()

            # Perform a backward pass to calculate gradients
            loss.backward()

            # Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update model parameters:
            # fine tune BERT params and train additional dense layers
            optimizer.step()
            # update learning rate
            scheduler.step()

            # Print the loss values and time elapsed for every 100 batches
            if (step % 100 == 0 and step != 0) or (step == len(train_dataloader) - 1):
                # Calculate time elapsed for 20 batches
                time_elapsed = time.time() - t0_batch

                print(f"{step:^9} | {batch_loss / batch_counts:^12.6f} | {time_elapsed:^9.2f}")

                # Reset batch tracking variables
                batch_loss, batch_counts = 0, 0
                t0_batch = time.time()

            # Calculate the average loss over the entire training data
        avg_train_loss = total_loss / len(train_dataloader)
        ###EVALUATION###

        # Put the model into the evaluation mode
        model.eval()

        # Define empty lists to host accuracy and validation for each batch
        val_accuracy = []
        val_loss = []

        for batch in val_dataloader:
            batch_input_ids, batch_attention_mask, batch_labels = tuple(t.to(DEVICE) for t in batch)

            # We do not want to update the params during the evaluation,
            # So we specify that we dont want to compute the gradients of the tensors
            # by calling the torch.no_grad() method
            with torch.no_grad():
                logits = model(batch_input_ids, batch_attention_mask)

            loss = loss_fn(logits, batch_labels)

            val_loss.append(loss.item())

            # Get the predictions starting from the logits (get index of highest logit)
            preds = torch.argmax(logits, dim=1).flatten()

            # Calculate the validation accuracy
            accuracy = (preds == batch_labels).cpu().numpy().mean() * 100
            val_accuracy.append(accuracy)

        # Compute the average accuracy and loss over the validation set
        val_loss = np.mean(val_loss)
        val_accuracy = np.mean(val_accuracy)

        # Print performance over the entire training data
        time_elapsed = time.time() - t0_epoch
        print("-" * 61)
        print(f"{'AVG TRAIN LOSS':^12} | {'VAL LOSS':^10} | {'VAL ACCURACY (%)':^9} | {'ELAPSED (s)':^9}")
        print("-" * 61)
        print(f"{avg_train_loss:^14.6f} | {val_loss:^10.6f} | {val_accuracy:^17.2f} | {time_elapsed:^9.2f}")
        print("-" * 61)
        print("\n")
    model.save_pretrained("./model")

BertTrain(bert_classifier, train_dataloader, val_dataloader, epochs=EPOCHS)
tokenizer.save_pretrained("./model")

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

        # Get index of highest logit
        pred = torch.argmax(logit, dim=1).cpu().numpy()
        # Append predicted class to list
        preds_list.extend(pred)

    return preds_list


bert_preds = bert_predict(bert_classifier, test_dataloader)
print('Classification Report for BERT :\n', classification_report(y_test, bert_preds, target_names=sentiments))
