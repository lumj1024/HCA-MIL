# -*- coding: utf-8 -*-
"""
test_full.py
"""
import sys
sys.path.insert(0, '/mnt/lmj/prcv2026')

import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
from matplotlib.patches import Ellipse

from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    precision_score, recall_score, f1_score,
    confusion_matrix,
    roc_curve, precision_recall_curve, average_precision_score,
)
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from dataset import make_data
from model.Groupmamba.models.groupmambav9 import groupmamba_tiny

# ============================================================
# ★ 修改这里 ★
# ============================================================
MODEL_PATH  = '/mnt/lmj/prcv2026/log/3407/checkpoint/epoch_142.pth'
SAVE_DIR    = '/mnt/lmj/prcv2026/test_results'
GPU         = 0
CLASS_NAMES = ['ALL', 'CLL']
# ============================================================

os.makedirs(SAVE_DIR, exist_ok=True)
device = torch.device(f'cuda:{GPU}' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# 加载模型
model = groupmamba_tiny(num_classes=2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()
print(f'Loaded: {MODEL_PATH}')

# 加载测试集
test_root_path = '/mnt/disk1/lumingjie/data/leukemiadata/test/'
test_seq_dir   = '/dev/shm/lmjdata/leukemiadata/test/'
test_dataset   = make_data(test_root_path, test_seq_dir,
                           seq=16, extension=1, generate=0, stage='test')
test_loader    = torch.utils.data.DataLoader(
    dataset=test_dataset, num_workers=8, batch_size=1, shuffle=False)
print(f'Test samples: {len(test_loader)}')

# 推理
all_labels, all_probs, all_preds = [], [], []
print('Running inference...')
with torch.no_grad():
    for _, images, labels in tqdm(test_loader, desc='Testing'):
        images = images.to(device).squeeze(0).permute(0, 3, 1, 2)
        labels = labels.to(device)
        logits = model(images)
        probs  = F.softmax(logits, dim=1)
        pred   = torch.argmax(probs, dim=1)
        all_labels.append(labels.item())
        all_probs.append(probs[0].cpu().numpy())
        all_preds.append(pred.item())

all_labels = np.array(all_labels)
all_probs  = np.array(all_probs)
all_preds  = np.array(all_preds)

# 基础指标
acc       = accuracy_score(all_labels, all_preds)
auc_macro = roc_auc_score(all_labels, all_probs[:, 1])
precision = precision_score(all_labels, all_preds, average='binary', zero_division=0)
recall    = recall_score(all_labels, all_preds, average='binary', zero_division=0)
f1        = f1_score(all_labels, all_preds, average='binary', zero_division=0)
cm        = confusion_matrix(all_labels, all_preds)
acc0      = cm[0, 0] / cm[0].sum()
acc1      = cm[1, 1] / cm[1].sum()

print(f'\n{"="*50}')
print(f'  ACC       : {acc*100:.2f}%')
print(f'  AUC       : {auc_macro:.4f}')
print(f'  Precision : {precision:.4f}')
print(f'  Recall    : {recall:.4f}')
print(f'  F1        : {f1:.4f}')
print(f'  Acc0(ALL) : {acc0*100:.2f}%')
print(f'  Acc1(CLL) : {acc1*100:.2f}%')
print(f'  Confusion Matrix:')
print(f'               Pred_ALL  Pred_CLL')
print(f'  True_ALL  :   {cm[0,0]:>5}     {cm[0,1]:>5}')
print(f'  True_CLL  :   {cm[1,0]:>5}     {cm[1,1]:>5}')
print(f'{"="*50}')

# 保存基础数据
pd.DataFrame({
    'label': all_labels, 'pred': all_preds,
    'prob_ALL': all_probs[:, 0], 'prob_CLL': all_probs[:, 1],
    'correct': (all_labels == all_preds).astype(int),
}).to_csv(os.path.join(SAVE_DIR, 'predictions.csv'), index=False)

pd.DataFrame([{
    'ACC': acc, 'AUC': auc_macro, 'Precision': precision,
    'Recall': recall, 'F1': f1, 'Acc0_ALL': acc0, 'Acc1_CLL': acc1,
}]).to_csv(os.path.join(SAVE_DIR, 'metrics.csv'), index=False)
print('[Saved] predictions.csv / metrics.csv')

# ============================================================
# 混淆矩阵
# ============================================================
fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(cm, cmap='Blues')
plt.colorbar(im, ax=ax)
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(CLASS_NAMES, fontsize=13)
ax.set_yticklabels(CLASS_NAMES, fontsize=13)
ax.set_xlabel('Predicted Label', fontsize=13)
ax.set_ylabel('True Label', fontsize=13)
ax.set_title('Confusion Matrix', fontsize=14)
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                color='white' if cm[i, j] > cm.max() / 2 else 'black',
                fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'confusion_matrix.png'), dpi=300)
plt.close()
pd.DataFrame(cm,
             index=[f'True_{n}' for n in CLASS_NAMES],
             columns=[f'Pred_{n}' for n in CLASS_NAMES]
             ).to_csv(os.path.join(SAVE_DIR, 'confusion_matrix.csv'))
print('[Saved] confusion_matrix.png / .csv')

# ============================================================
# ROC 曲线 — ALL / CLL / Macro Average
# ============================================================
print('\nComputing ROC curves...')
roc_styles = {
    'ALL':   dict(color='#2E86AB', lw=2, ls='-'),
    'CLL':   dict(color='#D85A30', lw=2, ls='-'),
    'Macro': dict(color='#444444', lw=2, ls='--'),
}
roc_data = {}

fig, ax = plt.subplots(figsize=(6, 6))
ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle=':', label='Random')

for cls_idx, cls_name in enumerate(CLASS_NAMES):
    y_true  = (all_labels == cls_idx).astype(int)
    y_score = all_probs[:, cls_idx]
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc_val = roc_auc_score(y_true, y_score)
    roc_data[cls_name] = {'FPR': fpr, 'TPR': tpr, 'AUC': auc_val}
    s = roc_styles[cls_name]
    ax.plot(fpr, tpr, color=s['color'], lw=s['lw'], ls=s['ls'],
            label=f'{cls_name} (AUC = {auc_val:.4f})')
    print(f'  ROC AUC [{cls_name}]: {auc_val:.4f}')

mean_fpr = np.linspace(0, 1, 200)
mean_tpr = np.mean([np.interp(mean_fpr, roc_data[n]['FPR'], roc_data[n]['TPR'])
                    for n in CLASS_NAMES], axis=0)
mean_tpr[0] = 0.0; mean_tpr[-1] = 1.0
auc_macro_cls = np.mean([roc_data[n]['AUC'] for n in CLASS_NAMES])
roc_data['Macro'] = {'FPR': mean_fpr, 'TPR': mean_tpr, 'AUC': auc_macro_cls}
s = roc_styles['Macro']
ax.plot(mean_fpr, mean_tpr, color=s['color'], lw=s['lw'], ls=s['ls'],
        label=f'Macro Avg (AUC = {auc_macro_cls:.4f})')
print(f'  ROC AUC [Macro]:  {auc_macro_cls:.4f}')

ax.set_xlim([0.0, 1.0]); ax.set_ylim([0.0, 1.02])
ax.set_xlabel('False Positive Rate', fontsize=13)
ax.set_ylabel('True Positive Rate', fontsize=13)
ax.set_title('ROC Curves', fontsize=14)
ax.legend(loc='lower right', fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'roc_curve.png'), dpi=300)
plt.close()

df_roc_dict = {}
auc_roc_dict = {}
for cls_name in list(CLASS_NAMES) + ['Macro']:
    df_roc_dict[f'FPR_{cls_name}'] = pd.Series(roc_data[cls_name]['FPR'])
    df_roc_dict[f'TPR_{cls_name}'] = pd.Series(roc_data[cls_name]['TPR'])
    auc_roc_dict[f'AUC_{cls_name}'] = roc_data[cls_name]['AUC']
pd.DataFrame(df_roc_dict).to_csv(os.path.join(SAVE_DIR, 'roc_curve.csv'), index=False)
pd.DataFrame([auc_roc_dict]).to_csv(os.path.join(SAVE_DIR, 'roc_auc_summary.csv'), index=False)
print('[Saved] roc_curve.png / roc_curve.csv / roc_auc_summary.csv')

# ============================================================
# PR 曲线 — ALL / CLL / Macro Average
# ============================================================
print('\nComputing PR curves...')
pr_styles = {
    'ALL':   dict(color='#2E86AB', lw=2, ls='-'),
    'CLL':   dict(color='#D85A30', lw=2, ls='-'),
    'Macro': dict(color='#444444', lw=2, ls='--'),
}
pr_data = {}

fig, ax = plt.subplots(figsize=(6, 6))

for cls_idx, cls_name in enumerate(CLASS_NAMES):
    y_true  = (all_labels == cls_idx).astype(int)
    y_score = all_probs[:, cls_idx]
    prec_vals, rec_vals, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    pr_data[cls_name] = {'Recall': rec_vals, 'Precision': prec_vals, 'AP': ap}
    s = pr_styles[cls_name]
    ax.plot(rec_vals, prec_vals, color=s['color'], lw=s['lw'], ls=s['ls'],
            label=f'{cls_name} (AP = {ap:.4f})')
    print(f'  PR AP [{cls_name}]: {ap:.4f}')

mean_rec = np.linspace(0, 1, 200)
mean_prec = np.mean([
    np.interp(mean_rec, pr_data[n]['Recall'][::-1], pr_data[n]['Precision'][::-1])
    for n in CLASS_NAMES], axis=0)
ap_macro = np.mean([pr_data[n]['AP'] for n in CLASS_NAMES])
pr_data['Macro'] = {'Recall': mean_rec, 'Precision': mean_prec, 'AP': ap_macro}
s = pr_styles['Macro']
ax.plot(mean_rec, mean_prec, color=s['color'], lw=s['lw'], ls=s['ls'],
        label=f'Macro Avg (AP = {ap_macro:.4f})')
print(f'  PR AP [Macro]:  {ap_macro:.4f}')

ax.set_xlim([0.0, 1.0]); ax.set_ylim([0.0, 1.02])
ax.set_xlabel('Recall', fontsize=13)
ax.set_ylabel('Precision', fontsize=13)
ax.set_title('Precision-Recall Curves', fontsize=14)
ax.legend(loc='lower left', fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'pr_curve.png'), dpi=300)
plt.close()

df_pr_dict = {}
ap_pr_dict = {}
for cls_name in list(CLASS_NAMES) + ['Macro']:
    df_pr_dict[f'Recall_{cls_name}']    = pd.Series(pr_data[cls_name]['Recall'])
    df_pr_dict[f'Precision_{cls_name}'] = pd.Series(pr_data[cls_name]['Precision'])
    ap_pr_dict[f'AP_{cls_name}'] = pr_data[cls_name]['AP']
pd.DataFrame(df_pr_dict).to_csv(os.path.join(SAVE_DIR, 'pr_curve.csv'), index=False)
pd.DataFrame([ap_pr_dict]).to_csv(os.path.join(SAVE_DIR, 'pr_ap_summary.csv'), index=False)
print('[Saved] pr_curve.png / pr_curve.csv / pr_ap_summary.csv')

# ============================================================
# t-SNE (PCA 预降维 + 置信椭圆)
# ============================================================
print('\nExtracting bag-level features for t-SNE...')

bag_features    = []
bag_labels_tsne = []

def hook_inter_fn(module, input, output):
    feat = input[0].detach().cpu().numpy()
    bag_features.append(feat.mean(axis=0))

hook2 = model.hca_inter.register_forward_hook(hook_inter_fn)
with torch.no_grad():
    for _, images, labels in tqdm(test_loader, desc='t-SNE features'):
        images = images.to(device).squeeze(0).permute(0, 3, 1, 2)
        _ = model(images)
        bag_labels_tsne.append(labels.item())
hook2.remove()

bag_features    = np.array(bag_features)
bag_labels_tsne = np.array(bag_labels_tsne)
n = len(bag_features)

# PCA 预降维
n_pca = min(50, n - 1, bag_features.shape[1])
pca = PCA(n_components=n_pca, random_state=42)
features_pca = pca.fit_transform(bag_features)
print(f'PCA explained variance: {pca.explained_variance_ratio_.sum():.3f}')

# t-SNE
print('Running t-SNE...')
tsne = TSNE(
    n_components=2,
    perplexity=max(5, min(n // 5, 30)),
    n_iter=2000,
    learning_rate='auto',
    init='pca',
    random_state=3407,
)
tsne_embed = tsne.fit_transform(features_pca)

# 置信椭圆
def confidence_ellipse(x, y, ax, n_std=1.5, **kwargs):
    if len(x) < 3:
        return
    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    rx = np.sqrt(1 + pearson)
    ry = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=rx * 2, height=ry * 2, **kwargs)
    scale_x = np.sqrt(cov[0, 0]) * n_std
    scale_y = np.sqrt(cov[1, 1]) * n_std
    transf = (transforms.Affine2D()
              .rotate_deg(45)
              .scale(scale_x, scale_y)
              .translate(np.mean(x), np.mean(y)))
    ellipse.set_transform(transf + ax.transData)
    ax.add_patch(ellipse)

colors  = {0: '#2E86AB', 1: '#D85A30'}
markers = {0: 'o',       1: 's'}

fig, ax = plt.subplots(figsize=(6, 5))
for cls_id, cls_name in enumerate(CLASS_NAMES):
    idx  = (bag_labels_tsne == cls_id)
    x_pts = tsne_embed[idx, 0]
    y_pts = tsne_embed[idx, 1]
    ax.scatter(x_pts, y_pts,
               c=colors[cls_id], marker=markers[cls_id],
               label=cls_name, s=80, alpha=0.9,
               edgecolors='white', linewidths=0.5)
    confidence_ellipse(x_pts, y_pts, ax, n_std=1.5,
                       facecolor=colors[cls_id], alpha=0.15,
                       edgecolor=colors[cls_id], linewidth=1.5)

ax.set_xlabel('t-SNE dim 1', fontsize=13)
ax.set_ylabel('t-SNE dim 2', fontsize=13)
ax.set_title('t-SNE of Bag-level Features', fontsize=14)
ax.legend(fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'tsne.png'), dpi=300)
plt.close()

for cls_id, cls_name in enumerate(CLASS_NAMES):
    idx = (bag_labels_tsne == cls_id)
    pd.DataFrame({
        'tsne_1': tsne_embed[idx, 0],
        'tsne_2': tsne_embed[idx, 1],
    }).to_csv(os.path.join(SAVE_DIR, f'tsne_{cls_name}.csv'), index=False)

pd.DataFrame({
    'tsne_1': tsne_embed[:, 0],
    'tsne_2': tsne_embed[:, 1],
    'label':  bag_labels_tsne,
    'class':  [CLASS_NAMES[l] for l in bag_labels_tsne],
}).to_csv(os.path.join(SAVE_DIR, 'tsne.csv'), index=False)
print('[Saved] tsne.png / tsne.csv / tsne_ALL.csv / tsne_CLL.csv')

print(f'\n{"="*55}')
print(f'All results saved to: {SAVE_DIR}')
print(f'{"="*55}')