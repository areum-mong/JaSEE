import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
try:
    import config
except:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    import config
# ================================================
# ?„ë¡œ?íŠ¸: ?‰ì? ?ì„¸ ë¶„ë¥˜ ëª¨ë¸ ê°œë°œ
# ?¨ê³„: 4?¨ê³„ - ìµœì¢… ëª¨ë¸
# ?¤ëª…: ìµœì¢… ê°€ì¤‘ì¹˜ ë°?ë©”í??°ì´???µí•© ?€??
# ?‘ì„±?? 2026.05.13
# ================================================
import torch
import torch.nn as nn
import os

# 1. Paths
save_path = str(config.RESULTS_DIR / "final_attention_mlp.pt")
load_weights_path = 'best_AttentionMLP_Tuned.pt'

# 2. Architecture (Must match saved weights)
class DynamicAttentionMLP(nn.Module):
    def __init__(self, input_dim, hidden_layers, dropout_rate):
        super(DynamicAttentionMLP, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=input_dim, num_heads=1, batch_first=True)
        layers = []
        in_dim = input_dim
        for h_dim in hidden_layers:
            layers.extend([nn.Linear(in_dim, h_dim), nn.BatchNorm1d(h_dim), nn.ReLU(), nn.Dropout(dropout_rate)])
            in_dim = h_dim
        layers.extend([nn.Linear(in_dim, 1), nn.Sigmoid()])
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        attn_out, _ = self.attn(x.unsqueeze(1), x.unsqueeze(1), x.unsqueeze(1))
        return self.net(attn_out.squeeze(1))

# 3. Features list
feature_list = [
    'ear_x', 'ear_y', 'sh_x', 'sh_y', 'hip_x', 'hip_y', 'knee_x', 'knee_y', 'ank_x', 'ank_y',
    'dist_ear_sh', 'dist_sh_hip', 'dist_hip_kn', 'dist_kn_ank', 
    'ang_sh_hip_kn', 'ang_hip_kn_ank'
]

# 4. Load & Save
model = DynamicAttentionMLP(input_dim=16, hidden_layers=[256, 128, 64], dropout_rate=0.42)
model.load_state_dict(torch.load(load_weights_path))

torch.save({
    'model_state_dict': model.state_dict(),
    'hyperparameters': {
        'learning_rate': 0.0031,
        'dropout': 0.42,
        'hidden_size': [256, 128, 64],
        'batch_size': 32,
        'weight_decay': 9.47e-05
    },
    'features': feature_list,
    'threshold': 0.40,
    'performance': {
        'recall': 0.9545,
        'f1': 0.9318,
        'auc': 0.9852,
        'diff': 0.0384
    }
}, save_path)

# 5. Output
file_size = os.path.getsize(save_path) / 1024
print(f"ëª¨ë¸ ?€???„ë£Œ: {save_path}")
print(f"ëª¨ë¸ ?Œì¼ ?¬ê¸°: {file_size:.2f} KB")


