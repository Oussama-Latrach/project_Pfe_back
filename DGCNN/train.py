import torch
import torch.optim as optim
import torch.nn as nn
from models.dgcnn import DGCNN
from utils.data_loader import get_loaders
from utils.metrics import compute_metrics
from utils.early_stopping import EarlyStopping
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Paramètres (peuvent être modifiés ici)
    num_classes = 4
    epochs = 10
    batch_size = 128
    learning_rate = 0.01
    weight_decay = 1e-3

    os.makedirs('experiments', exist_ok=True)
    os.makedirs('experiments/plots', exist_ok=True)

    # Initialisation des loaders et modèle
    train_loader, val_loader, test_loader = get_loaders(batch_size)
    model = DGCNN(num_classes=num_classes).to(device)

    # Optimizer et scheduler
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # Poids des classes et loss function
    class_counts = np.array([40688, 240847, 138346, 133705])
    class_weights = torch.tensor(1. / (class_counts / class_counts.sum()), dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Early stopping
    early_stopping = EarlyStopping(patience=5, mode='max')

    # Dictionnaire pour suivre les métriques
    metrics_history = {
        'train_loss': [],
        'train_acc': [],
        'val_acc': [],
        'val_f1_macro': [],
        'val_f1_weighted': []
    }

    def train_epoch(model, loader):
        model.train()
        total_loss, correct = 0, 0

        for data, target in loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (output.argmax(1) == target).sum().item()

        return total_loss / len(loader), correct / len(loader.dataset)

    def evaluate(model, loader):
        model.eval()
        all_preds, all_targets = [], []

        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                all_preds.extend(output.argmax(1).cpu().numpy())
                all_targets.extend(target.cpu().numpy())

        return compute_metrics(all_targets, all_preds)

    def save_plots(metrics, epoch):
        """Sauvegarde les graphiques des métriques"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Graphique de loss
        plt.figure()
        plt.plot(metrics['train_loss'], label='Train Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.legend()
        plt.savefig(f'experiments/plots/loss_{timestamp}.png')
        plt.close()

        # Graphique d'accuracy
        plt.figure()
        plt.plot(metrics['train_acc'], label='Train Accuracy')
        plt.plot(metrics['val_acc'], label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy Metrics')
        plt.legend()
        plt.savefig(f'experiments/plots/accuracy_{timestamp}.png')
        plt.close()

        # Graphique de F1 scores
        plt.figure()
        plt.plot(metrics['val_f1_macro'], label='F1 Macro')
        plt.plot(metrics['val_f1_weighted'], label='F1 Weighted')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.title('Validation F1 Scores')
        plt.legend()
        plt.savefig(f'experiments/plots/f1_scores_{timestamp}.png')
        plt.close()

    best_val_f1 = 0.0
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader)
        val_metrics = evaluate(model, val_loader)
        scheduler.step()

        # Mise à jour des métriques
        metrics_history['train_loss'].append(train_loss)
        metrics_history['train_acc'].append(train_acc)
        metrics_history['val_acc'].append(val_metrics['accuracy'])
        metrics_history['val_f1_macro'].append(val_metrics['f1_macro'])
        metrics_history['val_f1_weighted'].append(val_metrics['f1_weighted'])

        print(f"\nEpoch {epoch}:")
        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.2%}")
        print(
            f"Val Acc: {val_metrics['accuracy']:.2%} | F1 Macro: {val_metrics['f1_macro']:.4f} | F1 Weighted: {val_metrics['f1_weighted']:.4f}")

        # Sauvegarde des graphiques
        save_plots(metrics_history, epoch)

        # Early stopping check
        if early_stopping(val_metrics['f1_macro']):
            print(f"Early stopping triggered at epoch {epoch}")
            break

        # Sauvegarde du meilleur modèle
        if val_metrics['f1_macro'] > best_val_f1:
            best_val_f1 = val_metrics['f1_macro']
            torch.save(model.state_dict(), 'experiments/best_model.pth')

    torch.save(model.state_dict(), 'experiments/final_model.pth')


if __name__ == '__main__':
    main()