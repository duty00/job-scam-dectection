"""End-to-end TextCNN extension evaluated with the project's locked split."""
from collections import Counter
import json
import random
import re
import time

import numpy as np
import pandas as pd

from .data import save_json
from .evaluation import choose_threshold, metrics


TOKEN_PATTERN = re.compile(r"[a-z]+(?:'[a-z]+)?|urltoken|emailtoken")


def tokenize(text):
    """Tokenize already-cleaned English text without using labels or test statistics."""
    return TOKEN_PATTERN.findall(str(text).lower())


def build_vocabulary(texts, max_size=30000, min_frequency=2):
    """Fit a deterministic vocabulary on training text only."""
    counts = Counter(token for text in texts for token in tokenize(text))
    ordered = sorted(
        ((token, count) for token, count in counts.items() if count >= min_frequency),
        key=lambda item: (-item[1], item[0]),
    )[: max(0, max_size - 2)]
    vocab = {'<PAD>': 0, '<UNK>': 1}
    vocab.update({token: index + 2 for index, (token, _) in enumerate(ordered)})
    return vocab


def encode_texts(texts, vocab, max_length=300):
    """Encode and right-pad text using a vocabulary learned from train only."""
    encoded = np.zeros((len(texts), max_length), dtype=np.int64)
    unknown = vocab['<UNK>']
    for row, text in enumerate(texts):
        ids = [vocab.get(token, unknown) for token in tokenize(text)[:max_length]]
        if ids:
            encoded[row, :len(ids)] = ids
    return encoded


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            'PyTorch is required for TextCNN. In Colab it is normally preinstalled; '
            'otherwise run: python -m pip install -r requirements-deep-learning.txt'
        ) from exc
    return torch


def run_textcnn(ctx, epochs=6, batch_size=128, max_length=300,
                vocab_size=30000, embedding_dim=100, channels=96,
                learning_rate=1e-3, patience=2):
    """Train TextCNN on train, select epoch/threshold on validation, test once."""
    torch = _require_torch()
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    config, frame, ix, y = ctx['config'], ctx['df'], ctx['ix'], ctx['y']
    seed = int(config.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_texts = frame.iloc[ix['train']].text.tolist()
    vocab = build_vocabulary(train_texts, max_size=vocab_size, min_frequency=2)
    token_ids = encode_texts(frame.text.tolist(), vocab, max_length=max_length)
    np.save(config.path('features')/'textcnn_token_ids.npy', token_ids, allow_pickle=False)
    (config.path('features')/'textcnn_vocabulary.json').write_text(
        json.dumps(vocab, ensure_ascii=False, sort_keys=True), encoding='utf-8')

    class TextCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(len(vocab), embedding_dim, padding_idx=0)
            self.convolutions = nn.ModuleList(
                [nn.Conv1d(embedding_dim, channels, kernel_size=k) for k in (3, 4, 5)]
            )
            self.dropout = nn.Dropout(0.5)
            self.output = nn.Linear(channels * 3, 1)

        def forward(self, tokens):
            embedded = self.embedding(tokens).transpose(1, 2)
            pooled = [torch.relu(conv(embedded)).amax(dim=2) for conv in self.convolutions]
            return self.output(self.dropout(torch.cat(pooled, dim=1))).squeeze(1)

    def loader(indices, shuffle=False):
        dataset = TensorDataset(
            torch.from_numpy(token_ids[indices]),
            torch.from_numpy(y[indices].astype(np.float32)),
        )
        generator = torch.Generator().manual_seed(seed)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                          num_workers=0, generator=generator)

    train_loader = loader(ix['train'], shuffle=True)
    validation_loader = loader(ix['validation'])
    test_loader = loader(ix['test'])
    model = TextCNN().to(device)
    negatives = int((y[ix['train']] == 0).sum())
    positives = int((y[ix['train']] == 1).sum())
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([negatives / positives], dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    def predict(data_loader):
        model.eval()
        blocks = []
        with torch.no_grad():
            for tokens, _ in data_loader:
                logits = model(tokens.to(device))
                blocks.append(torch.sigmoid(logits).cpu().numpy())
        return np.concatenate(blocks)

    history = []
    best = None
    stale = 0
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for tokens, labels in train_loader:
            tokens, labels = tokens.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(tokens), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        validation_score = predict(validation_loader)
        threshold = choose_threshold(y[ix['validation']], validation_score)
        row = {'epoch': epoch, 'train_loss': float(np.mean(losses)), 'threshold': threshold,
               **metrics(y[ix['validation']], validation_score, threshold)}
        history.append(row)
        key = (row['f1_fraud'], row['average_precision'], -epoch)
        if best is None or key > best['key']:
            best = {
                'key': key,
                'epoch': epoch,
                'threshold': threshold,
                'validation': row,
                'state': {name: value.detach().cpu().clone()
                          for name, value in model.state_dict().items()},
            }
            stale = 0
        else:
            stale += 1
        print(
            f'TextCNN epoch {epoch}/{epochs}: loss={row["train_loss"]:.4f}, '
            f'val_f1={row["f1_fraud"]:.4f}, val_ap={row["average_precision"]:.4f}',
            flush=True,
        )
        if epoch >= 3 and stale >= patience:
            break

    model.load_state_dict(best['state'])
    test_score = predict(test_loader)
    test_result = metrics(y[ix['test']], test_score, best['threshold'])
    elapsed = time.perf_counter() - started
    params = {
        'architecture': 'TextCNN', 'embedding_trainable': True,
        'embedding_dim': embedding_dim, 'channels_per_kernel': channels,
        'kernel_sizes': [3, 4, 5], 'dropout': 0.5,
        'max_length': max_length, 'vocabulary_size': len(vocab),
        'batch_size': batch_size, 'learning_rate': learning_rate,
        'maximum_epochs': epochs, 'selected_epoch': best['epoch'],
        'selection': 'validation fraud F1; tie-break validation AP then earlier epoch',
        'threshold_source': 'validation', 'test_used_for_selection': False,
        'device': str(device), 'torch_version': torch.__version__,
        'seed': seed, 'elapsed_seconds': elapsed,
    }
    torch.save({'state_dict': best['state'], 'parameters': params},
               config.path('models')/'textcnn.pt')
    pd.DataFrame(history).to_csv(config.path('results')/'deep_learning_history.csv', index=False)
    score_table = pd.DataFrame([
        {'split': 'validation', 'family': 'textcnn', 'threshold': best['threshold'],
         **{key: value for key, value in best['validation'].items()
            if key not in {'epoch', 'train_loss', 'threshold'}}},
        {'split': 'test', 'family': 'textcnn', 'threshold': best['threshold'], **test_result},
    ])
    score_table.to_csv(config.path('results')/'deep_learning_scores.csv', index=False)
    save_json({'parameters': params, 'validation': best['validation'], 'test': test_result},
              config.path('results')/'deep_learning_summary.json')

    test_frame = pd.DataFrame({
        'job_id': frame.iloc[ix['test']].job_id.to_numpy(),
        'y_true': y[ix['test']], 'score': test_score,
        'prediction': (test_score >= best['threshold']).astype(int),
    })
    test_frame.to_csv(config.path('results')/'deep_learning_predictions.csv', index=False)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    history_frame = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    history_frame.plot(x='epoch', y='train_loss', marker='o', ax=axes[0], legend=False)
    history_frame.plot(x='epoch', y=['f1_fraud', 'average_precision'], marker='o', ax=axes[1])
    axes[0].set_title('TextCNN training loss')
    axes[1].set_title('Validation metrics')
    axes[1].set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(config.path('results/figures')/'05_textcnn_training.png', dpi=160)
    plt.close(fig)
    cm_fig, cm_ax = plt.subplots(figsize=(4.8, 4.2))
    ConfusionMatrixDisplay.from_predictions(
        y[ix['test']], test_frame.prediction, display_labels=['Legitimate', 'Fraudulent'],
        colorbar=False, ax=cm_ax, cmap='Purples')
    cm_ax.set_title('TextCNN (locked test)')
    cm_fig.tight_layout()
    cm_fig.savefig(config.path('results/figures')/'cm_textcnn.png', dpi=160)
    plt.close(cm_fig)
    return score_table
