import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_squared_error
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset
import os

DATA_PATH = 'tmp/ulist_vns_full.csv'
MODEL_SAVE_PATH = 'models/distilbert-vndb'
REGRESSION_MODEL_SAVE_PATH = 'models/distilbert-vndb-regression'

if not os.path.exists(MODEL_SAVE_PATH):
    os.makedirs(MODEL_SAVE_PATH)

if not os.path.exists(REGRESSION_MODEL_SAVE_PATH):
    os.makedirs(REGRESSION_MODEL_SAVE_PATH)

def clean_text(text):
    cleaned = re.sub(r"[^A-Za-z0-9.,!?;:'\"()\- ]+", " ", str(text))
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    words = re.findall(r'\b[a-zA-Z]+\b', cleaned)
    return cleaned, len(words)

def preprocess_data(df):
    df['clean_notes'], df['word_count'] = zip(*df['notes'].apply(clean_text))
    df = df[df['word_count'] >= 6].reset_index(drop=True)

    df['class'] = pd.cut(df['vote'], bins=[0, 40, 70, 100], labels=[0, 1, 2])
    # [0, 40] = 0, [41, 70] = 1, [71, 100] = 2
    df = df[['clean_notes', 'class', 'vote']]
    # drop duplicates
    df = df.drop_duplicates(subset=['clean_notes']).reset_index(drop=True)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['class'])
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

def create_datasets(df, use_cached=True):
    cached = use_cached and os.path.exists(os.path.join(MODEL_SAVE_PATH, 'tokenizer_config.json'))
    if cached:
        print("Tokenizer already exists, loading from", MODEL_SAVE_PATH)
        tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_SAVE_PATH)
    else:
        tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    
    train_df, val_df = preprocess_data(df)
    train_dataset = VndbDataset(train_df, tokenizer, label_mode='class')
    val_dataset = VndbDataset(val_df, tokenizer, label_mode='class')

    if not cached:
        tokenizer.save_pretrained(MODEL_SAVE_PATH)
        print("Tokenizer saved to:", MODEL_SAVE_PATH)
    
    return train_dataset, val_dataset

def create_regression_datasets(df):
    cached = os.path.exists(os.path.join(REGRESSION_MODEL_SAVE_PATH, 'tokenizer_config.json'))
    if cached:
        print("Tokenizer already exists, loading from", REGRESSION_MODEL_SAVE_PATH)
        tokenizer = DistilBertTokenizerFast.from_pretrained(REGRESSION_MODEL_SAVE_PATH)
    else:
        tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    
    train_df, val_df = preprocess_data(df)
    train_dataset = VndbDataset(train_df, tokenizer, label_mode='vote')
    val_dataset = VndbDataset(val_df, tokenizer, label_mode='vote')
    
    if not cached:
        tokenizer.save_pretrained(REGRESSION_MODEL_SAVE_PATH)
        print("Tokenizer saved to:", REGRESSION_MODEL_SAVE_PATH)
    
    return train_dataset, val_dataset

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    warmup_steps=100,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir='./logs',
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model='eval_loss',
)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    report = classification_report(labels, preds, output_dict=True)
    accuracy = report['accuracy']
    return {"accuracy": accuracy}

def compute_mse_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.squeeze(-1)
    mse = mean_squared_error(labels, preds)
    return {"mse": mse}

def train_model(train_dataset, val_dataset):
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()
    model.save_pretrained(MODEL_SAVE_PATH)
    print("Model saved to:", MODEL_SAVE_PATH)
    return trainer

def train_regression_model(train_dataset, val_dataset):
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=1, problem_type='regression')

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_mse_metrics
    )

    trainer.train()
    model.save_pretrained(REGRESSION_MODEL_SAVE_PATH)
    print("Model saved to:", REGRESSION_MODEL_SAVE_PATH)
    return trainer

def load_model():
    if os.path.exists(os.path.join(MODEL_SAVE_PATH, 'model.safetensors')) or os.path.exists(os.path.join(MODEL_SAVE_PATH, 'pytorch_model.bin')):
        model = DistilBertForSequenceClassification.from_pretrained(MODEL_SAVE_PATH)
        tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_SAVE_PATH)
        print("Model and tokenizer loaded from:", MODEL_SAVE_PATH)
        return model, tokenizer
    else:
        raise FileNotFoundError(f"Model not found at {MODEL_SAVE_PATH}")

def evaluate_model(trainer, val_dataset):
    eval_results = trainer.evaluate(eval_dataset=val_dataset)
    print("Evaluation results:", eval_results)

    predictions, labels, _ = trainer.predict(val_dataset)
    preds = predictions.argmax(-1)
    mse = mean_squared_error(labels, preds)
    print("Mean Squared Error:", mse)
    return eval_results, mse

def main(model_name='distilbert-vndb'):
    df = pd.read_csv(DATA_PATH)

    if model_name == 'distilbert-vndb':
        train_dataset, val_dataset = create_datasets(df)
        trainer = train_model(train_dataset, val_dataset)
    elif model_name == 'distilbert-vndb-regression':
        train_dataset, val_dataset = create_regression_datasets(df)
        trainer = train_regression_model(train_dataset, val_dataset)
    
    eval_results = trainer.evaluate()
    print("Final evaluation results:", eval_results)
    with open(os.path.join('results', f'{model_name}.txt'), 'w') as f:
        f.write(str(eval_results))

if __name__ == "__main__":
    main()
