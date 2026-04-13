import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math
import argparse

# 导入 25 维双脑模型
from src.model import Model

# ==========================================
# 1. Mock 环境包装器 (适配器模式)
# ==========================================
# 用于欺骗 model.pre_forward，使其认为收到了真实的 Env 对象
class MockProblemFeat:
    pass

class MockResetState:
    pass

# ==========================================
# 2. 数据集加载类
# ==========================================
class L2SegDataset(Dataset):
    def __init__(self, data_path):
        super().__init__()
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"❌ 找不到训练数据: {data_path}")
        print(f"📦 正在加载数据集: {data_path}...")
        self.data = torch.load(data_path)
        print(f"✅ 成功加载 {len(self.data)} 个局部修复样本！")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def custom_collate(batch):
    # 由于图结构大小可能不一，batch_size=1 是最稳妥的选择
    return batch

# ==========================================
# 3. 主训练逻辑
# ==========================================
def train(config_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 启动 L2Seg 训练流程，使用设备: {device}")

    # --- 读取 YAML 配置 ---
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    model_params = config["model_params"]
    env_params = config["env_params"]
    
    # 自动计算或校对 sqrt_embedding_dim
    if "sqrt_embedding_dim" not in model_params:
        model_params["sqrt_embedding_dim"] = math.sqrt(model_params["embedding_dim"])

    # 初始化 25 维模型
    model = Model(**model_params).to(device)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    nar_criterion = nn.BCEWithLogitsLoss()
    ar_criterion = nn.CrossEntropyLoss(ignore_index=model.PAD_TOKEN)

    # 加载数据
    data_path = "results/l2seg_dataset/l2seg_training_data.pt"
    dataset = L2SegDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=custom_collate, shuffle=True)

    epochs = 20
    save_dir = "results/l2seg_dataset/checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    print(f"🔥 开始训练 (配置文件: {config_path})\n")
    for epoch in range(epochs):
        total_loss, total_nar, total_ar = 0.0, 0.0, 0.0
        
        for batch_idx, batch in enumerate(dataloader):
            sample = batch[0]
            state_dict = sample["state_dict"]
            
            # 搬运基础标签
            nar_labels = torch.tensor(sample["nar_labels"], dtype=torch.float32, device=device).unsqueeze(0) 
            ar_sequences = torch.tensor(sample["ar_sequences"], dtype=torch.long, device=device).unsqueeze(0)

            # --- 组装 Mock 环境对象 ---
            mock_feat = MockProblemFeat()
            mock_feat.depot_xy = state_dict["depot_xy"].unsqueeze(0).to(device)
            mock_feat.node_xy = state_dict["node_xy"].unsqueeze(0).to(device)
            mock_feat.node_demand = state_dict["node_demand"].unsqueeze(0).to(device)

            reset_state = MockResetState()
            reset_state.problem_feat = mock_feat
            reset_state.tour_index = state_dict["tour_index"].unsqueeze(0).to(device)
            reset_state.neighbours = state_dict["neighbours"].unsqueeze(0).to(device)
            
            # 补全 25 维模型所需的 22 维附加特征 (8静态+14动态)
            N = state_dict["node_xy"].shape[0]
            reset_state.l2seg_static_feats = state_dict.get("l2seg_static_feats", torch.zeros(N, 8)).unsqueeze(0).to(device)
            reset_state.l2seg_dynamic_feats = state_dict.get("l2seg_dynamic_feats", torch.zeros(N, 14)).unsqueeze(0).to(device)
            reset_state.pad_mask = None 

            # --- 前向传播 ---
            optimizer.zero_grad()
            model.pre_forward(reset_state)

            # 1. NAR 损失
            nar_logits = model.nar_forward()
            # 自动对齐标签长度 (防御切片导致的维度不一)
            if nar_labels.shape[1] != nar_logits.shape[1]:
                nar_labels = torch.cat([nar_labels, torch.zeros(1, nar_logits.shape[1] - nar_labels.shape[1], device=device)], dim=1)
            loss_nar = nar_criterion(nar_logits, nar_labels[:, :nar_logits.shape[1]])

            # 2. AR 损失 (Teacher Forcing)
            ar_input = ar_sequences[:, :-1]
            ar_target = ar_sequences[:, 1:]
            ar_logits = model.ar_forward(ar_input)
            loss_ar = ar_criterion(ar_logits.reshape(-1, model.vocab_size), ar_target.reshape(-1))

            # --- 反向传播 ---
            loss = loss_nar + loss_ar
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            total_nar += loss_nar.item()
            total_ar += loss_ar.item()

        print(f" Epoch {epoch:03d} | Loss: {total_loss/len(dataloader):.4f} (NAR: {total_nar/len(dataloader):.4f}, AR: {total_ar/len(dataloader):.4f})")

        # 保存带参数定义的 Checkpoint
        torch.save({
            "model_state_dict": model.state_dict(),
            "model_params": model_params,
            "env_params": env_params
        }, f"{save_dir}/checkpoint-{epoch}.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/train/cvrp100.yaml')
    args = parser.parse_args()
    train(args.config)