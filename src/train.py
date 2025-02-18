from enum import IntEnum, StrEnum
import torch as tc
import torch.nn as nn
from transformers import DistilBertModel, DistilBertTokenizer, Trainer, TrainingArguments
from datasets import load_dataset, Dataset, DatasetDict
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

def compute_metrics(p):
    preds, labels = p
    preds = np.argmax(preds, axis=1)
    return { "accuracy": accuracy_score(labels, preds) }

MODEL_NAME = "dbmdz/distilbert-base-turkish-cased"
TRAIN_SPLIT = 0.8
SEED_DATA = 2398471
NUM_EPOCH = 200
LEARN_RATE = 1e-4
BATCH_SIZE = 32



tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)

class NewsClassifier(nn.Module):
    def __init__(self, num_labels=3):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained(MODEL_NAME)
        self.bert.requires_grad_(False)  # Freeze base model
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.classifier(outputs.last_hidden_state[:, 0, :])  # CLS token output
        result = { "logits": logits }
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.classifier.out_features), labels.view(-1))
            result["loss"] = loss
        return result

class Label(IntEnum):
    NEGATIVE = 0
    NEUTRAL = 1
    POSITIVE = 2

# Load dataset (Example: Replace with actual data)
df = pd.read_csv("data.csv")
num_negative = len(df[df["label"] == Label.NEGATIVE])
num_neutral = len(df[df["label"] == Label.NEUTRAL])
num_positive = len(df[df["label"] == Label.POSITIVE])
# Round up to have dataset sizes at hundreds: 100, 200, 300,...
num_data = (np.min([num_negative, num_neutral, num_positive]) // 100) * 100
num_train = int(TRAIN_SPLIT * num_data)
num_test = num_data - num_train

print(f"-> Dataset size : {len(df)}")
print(f"   Negative size: {num_negative}")
print(f"   Neutral size : {num_neutral}")
print(f"   Positive size: {num_positive}")
print(f"   Balanced size: {num_data}")
print(f"   Train size   : {num_train}")
print(f"   Test  size   : {num_test}")

# Shuffle dataset
data_neg = df[df["label"] == Label.NEGATIVE].to_numpy()[:num_data]
assert len(data_neg) == num_data
print(f"data_neg[0]: {data_neg[0]}")
data_neu = df[df["label"] == Label.NEUTRAL].to_numpy()[:num_data]
assert len(data_neu) == num_data
print(f"data_neu[0]: {data_neu[0]}")
data_pos = df[df["label"] == Label.POSITIVE].to_numpy()[:num_data]
assert len(data_pos) == num_data
print(f"data_pos[0]: {data_pos[0]}")

rng_data = np.random.default_rng(SEED_DATA)
shuffle_idx = rng_data.permutation(num_data)

print(f"Shuffling...")
data_neg = data_neg[shuffle_idx]
print(f"data_neg[0]: {data_neg[0]}")
data_neu = data_neu[shuffle_idx]
print(f"data_neu[0]: {data_neu[0]}")
data_pos = data_pos[shuffle_idx]
print(f"data_pos[0]: {data_pos[0]}")

assert num_data == num_train + num_test
data = {
    "neg_train": data_neg[:num_train], "neg_test": data_neg[num_train:],
    "neu_train": data_neu[:num_train], "neu_test": data_neu[num_train:],
    "pos_train": data_pos[:num_train], "pos_test": data_pos[num_train:],
}
print(f"-> Data prepared with")
print(f"   neg_train: {data['neg_train'].shape}, neg_test: {data['neg_test'].shape}")
print(f"   neu_train: {data['neu_train'].shape}, neu_test: {data['neu_test'].shape}")
print(f"   pos_train: {data['pos_train'].shape}, pos_test: {data['pos_test'].shape}")



class NewsDataset(tc.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # Tokenize the text
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,  # Add [CLS] and [SEP]
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'  # Return PyTorch tensors
        )

        # Return tokenized text and the label
        return {
            'input_ids': encoding['input_ids'].squeeze(0),  # Remove the extra batch dimension
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': tc.tensor(label, dtype=tc.long)  # Ensure label is in the correct type
        }

# Create dataset dict for HuggingFace
train_texts = list(data['neg_train'][:, 0]) + list(data['neu_train'][:, 0]) + list(data['pos_train'][:, 0])
train_labels = list(data['neg_train'][:, 1]) + list(data['neu_train'][:, 1]) + list(data['pos_train'][:, 1])

test_texts = list(data['neg_test'][:, 0]) + list(data['neu_test'][:, 0]) + list(data['pos_test'][:, 0])
test_labels = list(data['neg_test'][:, 1]) + list(data['neu_test'][:, 1]) + list(data['pos_test'][:, 1])

train_dataset = NewsDataset(train_texts, train_labels, tokenizer)
test_dataset = NewsDataset(test_texts, test_labels, tokenizer)

# train_dataset = Dataset.from_dict({'text': train_texts, 'label': train_labels})
# test_dataset = Dataset.from_dict({'text': test_texts, 'label': test_labels})

# dataset = DatasetDict({
    # 'train': train_dataset,
    # 'test': test_dataset
# })


# # Preprocessing function
# def tokenize_data(examples):
    # return tokenizer(examples["text"], padding="max_length", truncation=True)

# dataset = dataset.map(tokenize_data, batched=True)
# dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# Initialize model
model = NewsClassifier(num_labels=3)

# FIXME: remove me, freeze the params in the classifier model directly
# # Freeze all parameters except the classifier
# for param in model.bert.parameters():
    # param.requires_grad = False

# Define Trainer
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=LEARN_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=NUM_EPOCH,
    logging_dir="./logs",
    logging_steps=200,
    evaluation_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

# Train model
trainer.train()

model.eval()
model.to("cpu")

# Example inference function
def classify_news(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with tc.no_grad():
        result = model(**inputs)
        logits = result['logits']
    return tc.argmax(logits, dim=1).item()

# Test
print(f"Ekonomi kotuye gidiyor: {classify_news('Ekonomi kötüye gidiyor')}")  # Expected: negative
print("Expected: negative (0)")

exit(0)







#
# dataset = load_dataset("csv", data_files="data.csv")
# print()
# exit(0)
#
# # Preprocessing function
# def tokenize_data(examples):
    # return tokenizer(examples["text"], padding="max_length", truncation=True)
#
# dataset = dataset.map(tokenize_data, batched=True)
# dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
#
# # Initialize model
# model = NewsClassifier(num_labels=3)
#
# # Define Trainer
# training_args = TrainingArguments(
    # output_dir="./results", num_train_epochs=3, per_device_train_batch_size=8,
    # logging_dir="./logs", evaluation_strategy="epoch"
# )
# trainer = Trainer(model=model, args=training_args, train_dataset=dataset["train"])
#
# # Train model
# trainer.train()
#
# # Example inference function
# def classify_news(text):
    # inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    # with torch.no_grad():
        # logits = model(**inputs)
    # return torch.argmax(logits, dim=1).item()
#
# # Test
# print(classify_news("Ekonomi kötüye gidiyor"))  # Expected: negative
#
