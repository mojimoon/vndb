import os
import re
import csv
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import optuna
import torch.nn as nn
# import tensorflow as tf
# import tensorflow.keras.layers as layers
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_squared_error
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments

DATA_PATH = 'tmp/ulist_vns_full.csv'
MODEL_SAVE_PATH = 'models/distilbert-vndb'
REPORT_PATH = 'logs/hyperparameter_report.csv'

# plotting

def top_n_words(texts, n=30):
    from collections import Counter
    all_words = ' '.join(texts).split()
    counter = Counter(all_words)
    common_words = counter.most_common(n)
    words, counts = zip(*common_words)
    return list(words)

def plot_length_distribution(df):
    plt.figure(figsize=(8,4))
    sns.histplot(df['clean_notes'].apply(lambda x: len(x.split())), bins=30, kde=True)
    plt.title('Word Count Distribution')
    plt.xlabel('Word Count')
    plt.ylabel('Frequency')
    plt.savefig('img/word_count_distribution.png')

def plot_wordcloud(df, label=1, top_words=None):
    from wordcloud import WordCloud
    text = ' '.join(df[df['class']==label]['clean_notes'])

    if top_words:
        for word in top_words:
            text = text.replace(word, '')
        text = re.sub(r'\s+', ' ', text).strip()

    wc = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10,5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'WordCloud for class={label} (Excl. Common Terms)')
    plt.savefig(f'img/wordcloud_class_{label}_excl.png')

def plot_wordcloud_lengthy(df, label=1, min_length=50, top_words=None):
    from wordcloud import WordCloud
    text = ' '.join(df[(df['class']==label) & (df['clean_notes'].str.len() >= min_length)]['clean_notes'])

    if top_words:
        for word in top_words:
            text = text.replace(word, '')
        text = re.sub(r'\s+', ' ', text).strip()
    
    wc = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10,5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'WordCloud for class={label} with min length {min_length} (Excl. Common Terms)')
    plt.savefig(f'img/wordcloud_class_{label}_minlen_{min_length}_excl.png')

def plot_top_words(df, n=20):
    from collections import Counter
    all_words = ' '.join(df['clean_notes']).split()
    counter = Counter(all_words)
    common_words = counter.most_common(n)
    words, counts = zip(*common_words)
    plt.figure(figsize=(10,5))
    sns.barplot(x=list(words), y=list(counts))
    plt.xticks(rotation=45)
    plt.title(f'Top {n} Words')
    plt.xlabel('Word')
    plt.ylabel('Count')
    plt.savefig('img/top_words.png')

def plot_length_vs_vote_scatter(df):
    plt.figure(figsize=(10, 6))
    df['length'] = df['clean_notes'].apply(lambda x: len(x.split()))
    sns.scatterplot(data=df, x='length', y='vote', alpha=0.5, color='steelblue')
    plt.title('Length vs Vote')
    plt.xlabel('Word Count')
    plt.ylabel('Vote')
    plt.savefig('img/length_vs_vote_scatter.png')

def plot_top_words_vs_vote(df, n=10):
    # show top 10 words for each class
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer(max_features=1000, stop_words='english')
    X = vectorizer.fit_transform(df['clean_notes'])
    feature_names = vectorizer.get_feature_names_out()
    df['vote20'] = pd.cut(df['vote'], bins=[0, 20, 40, 60, 80, 100], labels=[0, 1, 2, 3, 4])
    nbin = 5
    _words = [X[df['vote20'].to_numpy() == i].sum(axis=0).A1 for i in range(nbin)]
    _top = [np.argsort(_words[i])[-n:][::-1] for i in range(nbin)]
    _top_words = [feature_names[_top[i]] for i in range(nbin)]
    _top_counts = [_words[i][_top[i]] for i in range(nbin)]

    plt.figure(figsize=(nbin * 4, 6))
    for i in range(nbin):
        plt.subplot(1, nbin, i + 1)
        sns.barplot(x=_top_words[i], y=_top_counts[i], palette='viridis')
        plt.title(f'Top Words for Vote Ranged {i * 20}-{(i + 1) * 20}')
        plt.xlabel('Words')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('img/top_words_vs_vote_excl.png') 

def plot_votes(df):
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
    plt.savefig('img/vote_distribution.png')

    print(df['vote'].describe())

# Begin DistilBERT

def clean_text(text):
    cleaned = re.sub(r"[^A-Za-z0-9.,!?;:'\"()\- ]+", " ", str(text))
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    words = re.findall(r'\b[a-zA-Z]+\b', cleaned)
    return cleaned, len(words)

def preprocess_data(df):
    df['clean_notes'], df['word_count'] = zip(*df['notes'].apply(clean_text))
    df = df[df['word_count'] >= 5].reset_index(drop=True)

    # df['class'] = pd.cut(df['vote'], bins=[0, 40, 70, 100], labels=[0, 1, 2])
    df['class'] = pd.cut(df['vote'], bins=[0, 69, 100], labels=[0, 1])
    df = df[['clean_notes', 'class', 'vote']]
    # drop duplicates
    df = df.drop_duplicates(subset=['clean_notes']).reset_index(drop=True)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=21474, stratify=df['class'])
    return train_df, val_df

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
    # model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3)
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
    
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

def plot_cm(predictions, labels, save_path, _title='Confusion Matrix'):
    from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
    cm = confusion_matrix(labels, predictions, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    total = tn + fp + fn + tp
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else 0.0 
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    f1 = f1_score(labels, predictions)
    acc = accuracy_score(labels, predictions)

    pred_sum = np.sum(cm, axis=0)
    label_sum = np.sum(cm, axis=1)

    cm_ext = np.zeros((3, 3), dtype=int)
    cm_ext[:2, :2] = cm
    # cm_ext[2, :2] = pred_sum
    # cm_ext[:2, 2] = label_sum
    # cm_ext[2, 2] = total

    cell_text = [
        [f"{tn}\nTNR={tnr:.2f}", f"{fp}\nFPR={fpr:.2f}", f"{label_sum[0]}"],
        [f"{fn}\nFNR={fnr:.2f}", f"{tp}\nTPR={tpr:.2f}", f"{label_sum[1]}"],
        [f"{pred_sum[0]}", f"{pred_sum[1]}", f"({total})\nF1={f1:.2f}\nAcc={acc:.2f}"]
    ]

    plt.figure(figsize=(6, 6))
    ax = sns.heatmap(cm_ext, annot=cell_text, fmt="", cbar=False, cmap="Blues", annot_kws={"size": 12})

    ax.set_xticklabels(['Pred 0', 'Pred 1', 'Total'], fontsize=12)
    ax.set_yticklabels(['Label 0', 'Label 1', 'Total'], fontsize=12, rotation=0)
    plt.title(_title, fontsize=16, weight="bold")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def evaluate_model(model, val_dataset, save_path=None, plt_title='Confusion Matrix'):
    trainer = Trainer(model=model)
    eval_results = trainer.evaluate(eval_dataset=val_dataset)
    print("Evaluation results:", eval_results)
    
    predictions, labels, _ = trainer.predict(val_dataset)
    preds = predictions.argmax(-1)

    if save_path is not None:
        plot_cm(preds, labels, save_path, _title=plt_title)
    
    report = classification_report(labels, preds, output_dict=True)
    print("Classification Report:\n", report)
    
    mse = mean_squared_error(labels, preds)
    print("Mean Squared Error:", mse)
    
    return report, mse

# Begin Baseline Transformer

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

    # df['class'] = pd.cut(df['vote'], bins=[0, 40, 70, 100], labels=[0, 1, 2])
    df['class'] = pd.cut(df['vote'], bins=[0, 69, 100], labels=[0, 1])
    df = df[['clean_notes', 'class', 'vote']]
    # drop duplicates
    df = df.drop_duplicates(subset=['clean_notes']).reset_index(drop=True)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=65535, stratify=df['class'])
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
    plot_cm(y_pred, y_true, save_path='img/transformer_confusion_matrix.png', _title='Transformer Confusion Matrix')
    return model

# Begin Transformer w/ token-and-position embeddings

# class CustomMultiHeadAttention(layers.Layer):
#     def __init__(self, embed_dim, num_heads=2, **kwargs):
#         super().__init__(**kwargs)
#         assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

#         self.embed_dim = embed_dim
#         self.num_heads = num_heads
#         self.projection_dim = embed_dim // num_heads

#         self.query_dense = layers.Dense(embed_dim)
#         self.key_dense = layers.Dense(embed_dim)
#         self.value_dense = layers.Dense(embed_dim)
#         self.combine_heads = layers.Dense(embed_dim)

#     def attention(self, query, key, value):
#         score = tf.matmul(query, key, transpose_b=True)
#         dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
#         scaled_score = score / tf.math.sqrt(dim_key)
#         weights = tf.nn.softmax(scaled_score, axis=-1)
#         output = tf.matmul(weights, value)
#         return output

#     def separate_heads(self, x, batch_size):
#         x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
#         return tf.transpose(x, perm=[0, 2, 1, 3])  # (batch, heads, seq_len, dim)

#     def call(self, inputs):
#         batch_size = tf.shape(inputs)[0]

#         query = self.query_dense(inputs)
#         key = self.key_dense(inputs)
#         value = self.value_dense(inputs)

#         query = self.separate_heads(query, batch_size)
#         key = self.separate_heads(key, batch_size)
#         value = self.separate_heads(value, batch_size)

#         attention_output = self.attention(query, key, value)

#         attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])
#         concat_attention = tf.reshape(attention_output, (batch_size, -1, self.embed_dim))
#         output = self.combine_heads(concat_attention)
#         return output

#     def get_config(self):
#         config = super().get_config()
#         config.update({
#             "embed_dim": self.embed_dim,
#             "num_heads": self.num_heads
#         })
#         return config

# class TransformerBlock(layers.Layer):
#     def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
#         super().__init__(**kwargs)
#         self.embed_dim = embed_dim
#         self.num_heads = num_heads
#         self.ff_dim = ff_dim
#         self.rate = rate

#         self.att = CustomMultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads)
#         self.ffn = tf.keras.Sequential(
#             [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
#         )
#         self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
#         self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
#         self.dropout1 = tf.keras.layers.Dropout(rate)
#         self.dropout2 = tf.keras.layers.Dropout(rate)

#     def call(self, inputs):
#         attn_output = self.att(inputs)
#         attn_output = self.dropout1(attn_output)
#         out1 = self.layernorm1(inputs + attn_output)
#         ffn_output = self.ffn(out1)
#         ffn_output = self.dropout2(ffn_output)
#         return self.layernorm2(out1 + ffn_output)

#     def get_config(self):
#         config = super().get_config()
#         config.update({
#             "embed_dim": self.embed_dim,
#             "num_heads": self.num_heads,
#             "ff_dim": self.ff_dim,
#             "rate": self.rate,
#         })
#         return config

# class TokenAndPositionEmbedding(tf.keras.layers.Layer):
#     def __init__(self, maxlen, vocab_size, embed_dim, **kwargs):
#         super().__init__(**kwargs)
#         self.maxlen = maxlen
#         self.vocab_size = vocab_size
#         self.embed_dim = embed_dim
#         self.token_emb = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
#         self.pos_emb = tf.keras.layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

#     def call(self, x):
#         maxlen = tf.shape(x)[-1]
#         positions = tf.range(start=0, limit=maxlen, delta=1)
#         positions = self.pos_emb(positions)
#         x = self.token_emb(x)
#         return x + positions

#     def get_config(self):
#         config = super().get_config().copy()
#         config.update({
#             "maxlen": self.maxlen,
#             "vocab_size": self.vocab_size,
#             "embed_dim": self.embed_dim
#         })
#         return config

# def Transformer(maxlen, vocab_size):
#     embed_dim = 32
#     num_heads = 2
#     ff_dim = 32

#     inputs = tf.keras.layers.Input(shape=(maxlen,))
#     embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
#     x = embedding_layer(inputs)
#     transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
#     x = transformer_block(x)
#     x = tf.keras.layers.GlobalAveragePooling1D()(x)
#     x = tf.keras.layers.Dropout(0.1)(x)
#     x = tf.keras.layers.Dense(20, activation="relu")(x)
#     x = tf.keras.layers.Dropout(0.1)(x)
#     outputs = tf.keras.layers.Dense(2, activation="softmax")(x)

#     model = tf.keras.Model(inputs=inputs, outputs=outputs)
#     return model

# def train_transformer_embedding(train_df, val_df, vocab_size=20000, maxlen=200):
#     from tensorflow.keras.preprocessing import text
#     from tensorflow.keras.preprocessing.sequence import pad_sequences

#     tokenizer = text.Tokenizer(num_words=vocab_size, oov_token="<OOV>")
#     tokenizer.fit_on_texts(train_df['clean_notes'].tolist() + val_df['clean_notes'].tolist())
    
#     trainX = tokenizer.texts_to_sequences(train_df['clean_notes'].tolist())
#     trainX = pad_sequences(trainX, maxlen=maxlen)
#     trainy = train_df['class'].astype(int).values # to numpy array

#     valX = tokenizer.texts_to_sequences(val_df['clean_notes'].tolist())
#     valX = pad_sequences(valX, maxlen=maxlen)
#     valy = val_df['class'].astype(int).values

#     model = Transformer(maxlen, vocab_size)
#     model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

#     checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
#         filepath='models/transformer_embedding.h5',
#         save_best_only=True,
#         monitor='val_loss',
#         mode='min',
#         verbose=1
#     )
    
#     history = model.fit(
#         trainX, trainy, batch_size=16, epochs=5, verbose=2,
#         validation_data=(valX, valy),
#         callbacks=[checkpoint_cb]
#     )

#     tokenizer_json = tokenizer.to_json()
#     with open('models/transformer_tokenizer.json', 'w') as f:
#         json.dump(tokenizer_json, f)
    
#     return model, tokenizer

# def evaluate_transformer_embedding(model, val_df, tokenizer, maxlen=200):
#     from tensorflow.keras.preprocessing.sequence import pad_sequences

#     valX = tokenizer.texts_to_sequences(val_df['clean_notes'].tolist())
#     valX = pad_sequences(valX, maxlen=maxlen)
#     valy = val_df['class'].astype(int).values

#     loss, accuracy = model.evaluate(valX, valy, verbose=2)
#     print(f"Validation Loss: {loss:.4f}, Validation Accuracy: {accuracy:.4f}")

#     predictions = model.predict(valX)
#     preds = predictions.argmax(axis=-1)

#     report = classification_report(valy, preds, output_dict=True)
#     print("Classification Report:\n", report)

#     plot_cm(preds, valy, save_path='img/transformer_embedding_confusion_matrix.png', _title='Transformer Embedding Confusion Matrix')
    
#     return report

# def load_transformer_embedding():
#     from tensorflow.keras.models import load_model
#     model = load_model("models/transformer_embedding.h5", custom_objects={
#         "TokenAndPositionEmbedding": TokenAndPositionEmbedding,
#         "TransformerBlock": TransformerBlock,
#         "CustomMultiHeadAttention": CustomMultiHeadAttention
#     })
#     return model

# Begin main functions

def train_and_evaluate():
    df = pd.read_csv(DATA_PATH)
    train_dataset, val_dataset = create_datasets(df)

    model = train_model(train_dataset, val_dataset)

    evaluate_model(model, val_dataset, save_path='img/distilbert_confusion_matrix.png', plt_title='DistilBERT Confusion Matrix')

def train_with_hyperparameter_search():
    df = pd.read_csv(DATA_PATH)
    train_dataset, val_dataset = create_datasets(df)

    hyperparameter_search(train_dataset, val_dataset)

def train_and_evaluate_transformer_baseline():
    df = pd.read_csv(DATA_PATH)
    train_df, val_df = preprocess_data_transformer(df)
    vocab = TextVocab(train_df['clean_notes'].tolist() + val_df['clean_notes'].tolist())
    train_dataset = TransformerDataset(train_df, vocab)
    val_dataset = TransformerDataset(val_df, vocab)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)
    model = train_transformer(train_loader, val_loader, vocab_size=len(vocab.itos))

# def train_and_evaluate_transformer_embedding():
#     df = pd.read_csv(DATA_PATH)
#     train_df, val_df = preprocess_data_transformer(df)
#     model, tokenizer = train_transformer_embedding(train_df, val_df)
#     evaluate_transformer_embedding(model, val_df, tokenizer)

def general_stat():
    df = pd.read_csv(DATA_PATH)
    # plot_votes(df)

    df['clean_notes'], df['word_count'] = zip(*df['notes'].apply(clean_text_transformer))
    df = df[df['word_count'] >= 5].reset_index(drop=True)
    df['class'] = pd.cut(df['vote'], bins=[0, 69, 100], labels=[0, 1])

    # remove short words
    df['clean_notes'] = df['clean_notes'].apply(lambda x: ' '.join([w for w in x.split() if len(w) > 2]))

    top_words = top_n_words(df['clean_notes'].tolist(), n=30)

    # plot_length_distribution(df)
    plot_wordcloud(df, label=1, top_words=top_words)
    plot_wordcloud(df, label=0, top_words=top_words)
    plot_wordcloud_lengthy(df, label=1, min_length=50, top_words=top_words)
    plot_wordcloud_lengthy(df, label=0, min_length=50, top_words=top_words)
    plot_top_words(df, n=30)
    # plot_length_vs_vote_scatter(df)
    plot_top_words_vs_vote(df, n=15)

if __name__ == "__main__":
    # train_and_evaluate()
    # train_and_evaluate_transformer_baseline()
    # train_and_evaluate_transformer_embedding()
    general_stat()

