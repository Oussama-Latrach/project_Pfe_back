import torch
import numpy as np
from utils.data_loader import get_loaders
from models.dgcnn import DGCNN
from utils.metrics import compute_metrics
import os

def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            all_preds.extend(output.argmax(1).cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    metrics = compute_metrics(all_targets, all_preds)

    print("\nEvaluation Metrics:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1 Macro: {metrics['f1_macro']:.4f}")
    print(f"F1 Weighted: {metrics['f1_weighted']:.4f}")
    print("\nF1 by class:")
    for i, f1 in enumerate(metrics['f1_by_class']):
        print(f"Class {i+1}: {f1:.4f}")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs('experiments', exist_ok=True)

    _, _, test_loader = get_loaders(batch_size=128)
    model = DGCNN(num_classes=4).to(device)

    model_path = 'experiments/best_model.pth'
    model.load_state_dict(torch.load(model_path, map_location=device))

    evaluate_model(model, test_loader, device)

if __name__ == '__main__':
    main()