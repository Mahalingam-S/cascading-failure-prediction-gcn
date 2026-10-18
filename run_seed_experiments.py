"""
run_seed_experiments.py: Evaluates LabGCN, LabGAT, and MLP baseline across
three random seeds (42, 123, 456) to measure statistical stability (mean ± std).
"""

import json
import torch
import numpy as np
from src.dataset import LabDatasetManager
from src.models import LabGCN, LabGAT, MLPBaseline
from src.train import train_model, compute_metrics
from src.evaluate import collect_model_predictions

dataset_mgr = LabDatasetManager('data')
data = dataset_mgr.create_or_load_dataset(cache_path='data/lab_cascade_dataset.pt')
train_data, val_data, test_data = data['train'], data['val'], data['test']
pos_weight = data['pos_weight']

seeds = [42, 123, 456]
experiments = {}

for s in seeds:
    torch.manual_seed(s)
    np.random.seed(s)
    print(f"\n---> Running Seed {s}...")
    
    # LabGCN
    gcn = LabGCN(10, 64, 3, dropout=0.25)
    train_res_gcn = train_model(gcn, f'LabGCN_s{s}', train_data, val_data, pos_weight, epochs=25, lr=0.006)
    y_true, y_prob_gcn, y_sev_gcn = collect_model_predictions(gcn, test_data, torch.device('cpu'))
    m_gcn = compute_metrics(y_true, y_prob_gcn)
    
    # MLP Baseline
    mlp = MLPBaseline(10, 64, 3, dropout=0.25)
    train_res_mlp = train_model(mlp, f'MLP_s{s}', train_data, val_data, pos_weight, epochs=25, lr=0.006)
    y_true, y_prob_mlp, y_sev_mlp = collect_model_predictions(mlp, test_data, torch.device('cpu'))
    m_mlp = compute_metrics(y_true, y_prob_mlp)
    
    experiments[str(s)] = {
        'GCN': {
            'metrics': m_gcn,
            'best_epoch': train_res_gcn['best_epoch'],
            'history': train_res_gcn['history']
        },
        'MLP': {
            'metrics': m_mlp,
            'best_epoch': train_res_mlp['best_epoch'],
            'history': train_res_mlp['history']
        }
    }

with open('results/seed_experiments.json', 'w') as f:
    json.dump(experiments, f, indent=2)

print("\n" + "=" * 60)
print("SEED EXPERIMENT SUMMARY")
print("=" * 60)
for m_name in ['GCN', 'MLP']:
    accs = [experiments[str(s)][m_name]['metrics']['accuracy'] for s in seeds]
    f1s = [experiments[str(s)][m_name]['metrics']['f1'] for s in seeds]
    aucs = [experiments[str(s)][m_name]['metrics']['roc_auc'] for s in seeds]
    print(f"{m_name}:")
    for s in seeds:
        print(f"  Seed {s}: Acc={experiments[str(s)][m_name]['metrics']['accuracy']:.4f}, F1={experiments[str(s)][m_name]['metrics']['f1']:.4f}, AUC={experiments[str(s)][m_name]['metrics']['roc_auc']:.4f}")
    print(f"  Mean +/- Std: Acc={np.mean(accs):.4f} +/- {np.std(accs):.4f}, F1={np.mean(f1s):.4f} +/- {np.std(f1s):.4f}, AUC={np.mean(aucs):.4f} +/- {np.std(aucs):.4f}")
