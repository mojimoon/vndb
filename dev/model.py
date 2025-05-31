import os
import re
import csv
import numpy as np
import pandas as pd

import torch
import optuna
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_squared_error
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments

DATA_PATH = 'tmp/ulist_vns_full.csv'
MODEL_SAVE_PATH = 'models/distilbert-vndb'
# MODEL_SAVE_PATH = 'models/distilbert-vndb-2'
REPORT_PATH = 'logs/hyperparameter_report.csv'

# Begin DistilBERT

def clean_text(text):
    cleaned = re.sub(r"[^A-Za-z0-9.,!?;:'\"()\- ]+", " ", str(text))
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    words = re.findall(r'\b[a-zA-Z]+\b', cleaned)
    return cleaned, len(words)

def preprocess_data(df):
    df['clean_notes'], df['word_count'] = zip(*df['notes'].apply(clean_text))
    df = df[df['word_count'] >= 5].reset_index(drop=True)

    df['class'] = pd.cut(df['vote'], bins=[0, 40, 70, 100], labels=[0, 1, 2])
    # [0, 40] = 0, [41, 70] = 1, [71, 100] = 2
    # df['class'] = pd.cut(df['vote'], bins=[0, 64, 100], labels=[0, 1])
    df = df[['clean_notes', 'class', 'vote']]
    # drop duplicates
    df = df.drop_duplicates(subset=['clean_notes']).reset_index(drop=True)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42949, stratify=df['class'])
    return train_df, val_df

def plot_votes(df):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    x = np.arange(0, 101, 5)
    sns.histplot(df['vote'], bins=x, kde=False, color='steelblue', stat='count')

    # normal distribution line
    mu, std = df['vote'].mean(), df['vote'].std()
    p = np.exp(-0.5 * ((x - mu) / std) ** 2) / (std * np.sqrt(2 * np.pi))
    plt.plot(x, p * len(df) * (x[1] - x[0]), color='darkblue', label='Norm Dist')
    plt.legend()

    q1 = df['vote'].quantile(0.25)
    med = df['vote'].quantile(0.5)
    q3 = df['vote'].quantile(0.75)

    plt.axvline(q1, color='tab:orange', linestyle='--')
    plt.axvline(med, color='tab:purple', linestyle='--')
    plt.axvline(q3, color='tab:green', linestyle='--')

    _y = plt.ylim()[1]

    plt.text(q1 + 1, _y, f'Q1: {q1:.0f}', color='tab:orange')
    plt.text(med + 1, _y, f'Med: {med:.0f}', color='tab:purple')
    plt.text(q3 + 1, _y, f'Q3: {q3:.0f}', color='tab:green')
    plt.text(mu + 1, _y * 0.96, f'Mean: {mu:.1f}', color='tab:brown')

    plt.title('Vote Distribution')
    plt.xlabel('Vote')
    plt.ylabel('Frequency')
    plt.savefig('vote_distribution.png')

    print(df['vote'].describe())

class VndbDataset(Dataset):
    def __init__(self, df, tokenizer, label_mode='class', max_length=256):
        self.texts = df['clean_notes'].tolist()
        self.label_mode = label_mode
        if label_mode == 'class':
            self.labels = df['class'].astype(int).tolist()
        else:
            self.labels = df['vote'].astype(float).tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

def create_datasets(df):
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    train_df, val_df = preprocess_data(df)
    
    train_dataset = VndbDataset(train_df, tokenizer, label_mode='class')
    val_dataset = VndbDataset(val_df, tokenizer, label_mode='class')
    
    return train_dataset, val_dataset

training_args = TrainingArguments(
    output_dir='./checkpoints',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    warmup_steps=100,
    evaluation_strategy="steps",
    eval_steps=500,
    logging_dir='./logs',
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model='eval_loss',
)

def train_model(train_dataset, val_dataset):
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3)
    # model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=lambda p: {
            'accuracy': (p.predictions.argmax(-1) == p.label_ids).mean(),
            'f1': classification_report(p.label_ids, p.predictions.argmax(-1), output_dict=True)['weighted avg']['f1-score']
        }
    )
    
    trainer.train()
    model.save_pretrained(MODEL_SAVE_PATH)
    print("Model saved to:", MODEL_SAVE_PATH)
    return model

def hp_space(trial):
    return {
        "learning_rate": trial.suggest_loguniform("learning_rate", 5e-6, 1e-4),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [8, 16, 32]),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 2, 5),
        "warmup_steps": trial.suggest_categorical("warmup_steps", [0, 100, 200, 300]),
        "weight_decay": trial.suggest_loguniform("weight_decay", 1e-5, 1e-2),
    }

def csv_callback(study, trial):
    fieldnames = ['number', 'lr', 'batch_size', 'epochs', 'warmup_steps', 'weight_decay', 'value']
    with open(REPORT_PATH, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if f.tell() == 0:
            writer.writeheader()
        writer.writerow({
            'number': trial.number,
            'lr': trial.params['learning_rate'],
            'batch_size': trial.params['per_device_train_batch_size'],
            'epochs': trial.params['num_train_epochs'],
            'warmup_steps': trial.params['warmup_steps'],
            'weight_decay': trial.params['weight_decay'],
            'value': trial.value
        })

def hyperparameter_search(train_dataset, val_dataset):
    trainer = Trainer(
        model_init=lambda: DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3),
        args=TrainingArguments(
            output_dir="./checkpoints",
            evaluation_strategy="steps",
            eval_steps=500,
            logging_dir="./logs",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
        ),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=lambda p: {
            'eval_accuracy': (p.predictions.argmax(-1) == p.label_ids).mean(),
            'eval_f1': classification_report(p.label_ids, p.predictions.argmax(-1), output_dict=True)['weighted avg']['f1-score']
        }
    )

    best_run = trainer.hyperparameter_search(
        direction="maximize",
        hp_space=hp_space,
        n_trials=10,
        backend="optuna",
        compute_objective=lambda metrics: metrics['eval_f1'],
        # callbacks=[csv_callback]
    )

    df = best_run.study.trials_dataframe()
    df.to_csv(REPORT_PATH, index=False)

    print("Best run:", best_run)

def load_model():
    if os.path.exists(MODEL_SAVE_PATH):
        model = DistilBertForSequenceClassification.from_pretrained(MODEL_SAVE_PATH)
        return model
    else:
        raise FileNotFoundError(f"Model not found at {MODEL_SAVE_PATH}")

def evaluate_model(model, val_dataset):
    trainer = Trainer(model=model)
    eval_results = trainer.evaluate(eval_dataset=val_dataset)
    print("Evaluation results:", eval_results)
    
    predictions, labels, _ = trainer.predict(val_dataset)
    preds = predictions.argmax(-1)
    
    report = classification_report(labels, preds, output_dict=True)
    print("Classification Report:\n", report)
    
    mse = mean_squared_error(labels, preds)
    print("Mean Squared Error:", mse)
    
    return report, mse

# Begin Transformer

def download_nltk_resources():
    import nltk
    nltk.download('stopwords')
    nltk.download('wordnet')

def clean_text_transformer(text):
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    cleaned = re.sub(r"[^A-Za-z0-9.,!?;:'\"()\- ]+", " ", str(text))
    cleaned = re.sub(r'\s+', ' ', cleaned).strip().lower()
    tokens = cleaned.split()
    tokens = [w for w in tokens if w not in stop_words]
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return ' '.join(tokens), len(tokens)

def preprocess_data_transformer(df):
    df['clean_notes'], df['word_count'] = zip(*df['notes'].apply(clean_text_transformer))
    df = df[df['word_count'] >= 5].reset_index(drop=True)

    df['class'] = pd.cut(df['vote'], bins=[0, 40, 70, 100], labels=[0, 1, 2])
    df = df[['clean_notes', 'class', 'vote']]
    # drop duplicates
    df = df.drop_duplicates(subset=['clean_notes']).reset_index(drop=True)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42949, stratify=df['class'])
    return train_df, val_df

class TextVocab:
    def __init__(self, texts, min_freq=2):
        self.itos = ['<pad>', '<unk>']
        self.stoi = {t: i for i, t in enumerate(self.itos)}
        freq = {}
        for text in texts:
            for w in text.lower().split():
                freq[w] = freq.get(w, 0) + 1
        for word, cnt in freq.items():
            if cnt >= min_freq:
                self.stoi.setdefault(word, len(self.itos))
                self.itos.append(word)
    def encode(self, text, max_len):
        tokens = text.lower().split()
        ids = [self.stoi.get(w, self.stoi['<unk>']) for w in tokens]
        if len(ids) < max_len:
            ids += [self.stoi['<pad>']] * (max_len - len(ids))
        else:
            ids = ids[:max_len]
        return ids

class TransformerDataset(Dataset):
    def __init__(self, df, vocab, max_len=64):
        self.texts = df['clean_notes'].tolist()
        self.labels = df['class'].astype(int).tolist()
        self.vocab = vocab
        self.max_len = max_len
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        ids = torch.tensor(self.vocab.encode(self.texts[idx], self.max_len))
        label = torch.tensor(self.labels[idx])
        return ids, label

class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=2, num_classes=3, max_len=64, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=256, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, num_classes)
    def forward(self, x):
        # x: [batch, seq]
        emb = self.embedding(x) + self.pos_embedding[:, :x.size(1), :]
        emb = emb.transpose(0, 1)  # [seq, batch, d_model]
        out = self.transformer(emb)  # [seq, batch, d_model]
        out = out.mean(dim=0)  # [batch, d_model]
        logits = self.fc(out)
        return logits

def train_transformer(train_loader, val_loader, vocab_size, device='cuda'):
    model = TransformerClassifier(vocab_size=vocab_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0
    for epoch in range(8):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                preds = logits.argmax(-1).cpu().numpy()
                y_pred.extend(preds)
                y_true.extend(y.cpu().numpy())
        acc = np.mean(np.array(y_pred) == np.array(y_true))
        print(f"Epoch {epoch+1} val acc: {acc:.4f}")
        if acc > best_val_acc:
            best_val_acc = acc
            torch.save(model.state_dict(), 'models/transformer-baseline.pt')
    print(classification_report(y_true, y_pred))
    return model

# Begin main functions

def train_and_evaluate():
    df = pd.read_csv(DATA_PATH)
    train_dataset, val_dataset = create_datasets(df)

    model = train_model(train_dataset, val_dataset)

    evaluate_model(model, val_dataset)

def train_with_hyperparameter_search():
    df = pd.read_csv(DATA_PATH)
    train_dataset, val_dataset = create_datasets(df)

    hyperparameter_search(train_dataset, val_dataset)

def train_and_evaluate_transformer():
    df = pd.read_csv(DATA_PATH)
    train_df, val_df = preprocess_data_transformer(df)
    vocab = TextVocab(train_df['clean_notes'].tolist() + val_df['clean_notes'].tolist())
    train_dataset = TransformerDataset(train_df, vocab)
    val_dataset = TransformerDataset(val_df, vocab)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)
    model = train_transformer(train_loader, val_loader, vocab_size=len(vocab.itos))

if __name__ == "__main__":
    plot_votes(df=pd.read_csv(DATA_PATH))

