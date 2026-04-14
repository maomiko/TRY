import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math
import argparse
import random
import numpy as np
import csv

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
        self.data = torch.load(data_path, map_location="cpu", weights_only=True)
        print(f"✅ 成功加载 {len(self.data)} 个局部修复样本！")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def custom_collate(batch):
    # 图结构可能不一致，保留为样本列表，后续逐样本计算 loss
    return batch


def _read_trainer_params(config):
    trainer_params = config.get("trainer_params", {})
    nar_pos_weight = trainer_params.get("nar_pos_weight", None)
    if nar_pos_weight is None:
        # 向后兼容：旧配置若只设置 nar_loss_weight，则沿用为 pos_weight
        nar_pos_weight = float(trainer_params.get("nar_loss_weight", 1.0))
    return {
        "epochs": int(trainer_params.get("epochs", 20)),
        "learning_rate": float(trainer_params.get("learning_rate", 1e-4)),
        "batch_size": int(trainer_params.get("batch_size", 1)),
        "shuffle": bool(trainer_params.get("shuffle", True)),
        "num_workers": int(trainer_params.get("num_workers", 0)),
        "grad_clip_norm": float(trainer_params.get("grad_clip_norm", 1.0)),
        "nar_loss_weight": float(trainer_params.get("nar_loss_weight", 1.0)),
        "nar_pos_weight": float(nar_pos_weight),
        "ar_loss_weight": float(trainer_params.get("ar_loss_weight", 1.0)),
        "train_data_path": trainer_params.get("train_data_path", "results/l2seg_dataset/l2seg_training_data.pt"),
        "checkpoint_dir": trainer_params.get("checkpoint_dir", "results/l2seg_dataset/checkpoints"),
        "checkpoint_every": int(trainer_params.get("checkpoint_every", 1)),
        "metrics_csv": trainer_params.get("metrics_csv", "results/l2seg_dataset/train_metrics.csv"),
    }

# ==========================================
# 3. 主训练逻辑
# ==========================================
def train(config_path, seed=1234):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 启动 L2Seg 训练流程，使用设备: {device}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # --- 读取 YAML 配置 ---
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    model_params = config["model_params"]
    env_params = config["env_params"]
    trainer_params = _read_trainer_params(config)
    
    # 自动计算或校对 sqrt_embedding_dim
    if "sqrt_embedding_dim" not in model_params:
        model_params["sqrt_embedding_dim"] = math.sqrt(model_params["embedding_dim"])

    # 初始化 25 维模型
    model = Model(**model_params).to(device)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=trainer_params["learning_rate"])
    nar_pos_weight = torch.tensor([trainer_params["nar_pos_weight"]], device=device)
    nar_criterion = nn.BCEWithLogitsLoss(pos_weight=nar_pos_weight)
    ar_criterion = nn.CrossEntropyLoss(ignore_index=model.PAD_TOKEN)

    # 加载数据
    data_path = trainer_params["train_data_path"]
    dataset = L2SegDataset(data_path)
    dataloader = DataLoader(
        dataset,
        batch_size=trainer_params["batch_size"],
        collate_fn=custom_collate,
        shuffle=trainer_params["shuffle"],
        num_workers=trainer_params["num_workers"],
    )

    epochs = trainer_params["epochs"]
    save_dir = trainer_params["checkpoint_dir"]
    os.makedirs(save_dir, exist_ok=True)
    metrics_dir = os.path.dirname(trainer_params["metrics_csv"])
    if metrics_dir:
        os.makedirs(metrics_dir, exist_ok=True)

    print(f"🔥 开始训练 (配置文件: {config_path})\n")
    if not os.path.exists(trainer_params["metrics_csv"]):
        with open(trainer_params["metrics_csv"], "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "weighted_total_loss", "unweighted_loss_nar", "unweighted_loss_ar"])

    for epoch in range(epochs):
        total_loss, total_nar, total_ar, sample_count = 0.0, 0.0, 0.0, 0
        skipped_short_sequences = 0
        padded_nar_labels = 0
        truncated_nar_labels = 0
        
        for batch_idx, batch in enumerate(dataloader):
            optimizer.zero_grad()
            valid_items = 0
            losses = []

            for sample in batch:
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
                model.pre_forward(reset_state)

                # 1. NAR 损失
                nar_logits = model.nar_forward()
                # 自动对齐标签长度 (防御切片导致的维度不一)
                aligned_nar_labels = nar_labels
                if nar_labels.shape[1] < nar_logits.shape[1]:
                    aligned_nar_labels = torch.cat(
                        [nar_labels, torch.zeros(1, nar_logits.shape[1] - nar_labels.shape[1], device=device)],
                        dim=1
                    )
                    padded_nar_labels += 1
                elif nar_labels.shape[1] > nar_logits.shape[1]:
                    aligned_nar_labels = nar_labels[:, :nar_logits.shape[1]]
                    truncated_nar_labels += 1
                loss_nar = nar_criterion(nar_logits, aligned_nar_labels)

                # 2. AR 损失 (Teacher Forcing)
                if ar_sequences.shape[1] < 2:
                    skipped_short_sequences += 1
                    continue
                ar_input = ar_sequences[:, :-1]
                ar_target = ar_sequences[:, 1:]
                ar_logits = model.ar_forward(ar_input)
                loss_ar = ar_criterion(ar_logits.reshape(-1, model.vocab_size), ar_target.reshape(-1))

                # --- 汇总损失 ---
                loss = trainer_params["nar_loss_weight"] * loss_nar + trainer_params["ar_loss_weight"] * loss_ar
                valid_items += 1
                losses.append(loss)

                total_loss += loss.item()
                total_nar += loss_nar.item()
                total_ar += loss_ar.item()
                sample_count += 1

            if valid_items == 0:
                continue

            batch_loss = torch.stack(losses).mean()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=trainer_params["grad_clip_norm"])
            optimizer.step()

        if sample_count == 0:
            print(f" Epoch {epoch:03d} | 无有效样本，跳过保存")
            continue

        avg_loss = total_loss / sample_count
        avg_nar = total_nar / sample_count
        avg_ar = total_ar / sample_count
        print(f" Epoch {epoch:03d} | Loss: {avg_loss:.4f} (NAR: {avg_nar:.4f}, AR: {avg_ar:.4f})")
        if skipped_short_sequences > 0:
            print(f"  ⚠️ 跳过 {skipped_short_sequences} 条过短 AR 序列样本")
        if padded_nar_labels > 0:
            print(f"  ⚠️ 补齐 {padded_nar_labels} 条 NAR 标签（标签长度 < logits 长度）")
        if truncated_nar_labels > 0:
            print(f"  ⚠️ 截断 {truncated_nar_labels} 条 NAR 标签（标签长度 > logits 长度）")

        with open(trainer_params["metrics_csv"], "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, avg_loss, avg_nar, avg_ar])

        # 保存带参数定义的 Checkpoint
        if (epoch + 1) % trainer_params["checkpoint_every"] == 0 or (epoch + 1) == epochs:
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_params": model_params,
                "env_params": env_params,
                "trainer_params": trainer_params,
            }, f"{save_dir}/checkpoint-{epoch}.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/reproduce/train_cvrp100.yaml')
    parser.add_argument('--seed', type=int, default=1234)
    args = parser.parse_args()
    train(args.config, seed=args.seed)
