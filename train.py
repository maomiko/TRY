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
import sys

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


def _remap_ar_sequence(
    raw_ar_sequence,
    global_node_indices,
    global_problem_size,
    pad_token,
):
    """
    Remap AR tokens from global customer IDs to local subproblem IDs.

    Args:
        raw_ar_sequence: Original AR token sequence from label generation.
        global_node_indices: Local subproblem customers in global ID space.
        global_problem_size: Global problem size used to derive global END token.
        pad_token: PAD token index used by the model.

    Returns:
        List[int]: Safe local token sequence clamped to [0, pad_token].
    """
    local_num_customers = len(global_node_indices)
    g2l_map = {int(g_id): local_idx + 1 for local_idx, g_id in enumerate(global_node_indices)}
    g2l_map[0] = 0
    global_end_token = int(global_problem_size) + 1
    local_end_token = local_num_customers + 1

    mapped = []
    for token in raw_ar_sequence:
        token = int(token)
        if token in g2l_map:
            mapped.append(g2l_map[token])
        elif token == global_end_token:
            mapped.append(local_end_token)
        else:
            mapped.append(int(pad_token))

    return [max(0, min(int(t), int(pad_token))) for t in mapped]


def _remap_nar_labels(
    raw_nar_labels,
    global_node_indices,
    global_problem_size,
):
    """
    Normalize NAR labels into local subproblem node space: [depot, customers...].
    """
    safe_raw = [float(x) for x in raw_nar_labels]
    safe_global_nodes = [
        int(x) for x in global_node_indices if 0 < int(x) <= int(global_problem_size)
    ]
    expected_local_len = len(safe_global_nodes) + 1

    if len(safe_raw) == expected_local_len:
        return safe_raw, "local_aligned"
    if len(safe_raw) == len(safe_global_nodes):
        return [0.0] + safe_raw, "local_missing_depot"

    global_with_depot_len = int(global_problem_size) + 1
    # 兼容旧数据：标签仍在全局空间（长度通常为 problem_size+1 或更长）
    if len(safe_raw) >= global_with_depot_len:
        remapped = [safe_raw[0]]
        for node_id in safe_global_nodes:
            remapped.append(safe_raw[node_id] if node_id < len(safe_raw) else 0.0)
        return remapped, "global_to_local"

    return safe_raw, "unknown"


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
        "ar_insert_weight": float(trainer_params.get("ar_insert_weight", 1.0)),
        "ar_delete_weight": float(trainer_params.get("ar_delete_weight", 1.0)),
        "train_data_path": trainer_params.get("train_data_path", "results/l2seg_dataset/l2seg_training_data.pt"),
        "checkpoint_dir": trainer_params.get("checkpoint_dir", "results/l2seg_dataset/checkpoints"),
        "checkpoint_every": int(trainer_params.get("checkpoint_every", 1)),
        "metrics_csv": trainer_params.get("metrics_csv", "results/l2seg_dataset/train_metrics.csv"),
    }


def _validate_train_preflight(config_path, config, trainer_params):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"❌ 找不到配置文件: {config_path}")
    if "model_params" not in config or "env_params" not in config:
        raise ValueError("配置缺少 model_params 或 env_params。")
    data_path = str(trainer_params["train_data_path"]).strip()
    if not data_path:
        raise ValueError("trainer_params.train_data_path 不能为空。")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"❌ 找不到训练数据: {data_path}")
    if int(trainer_params["epochs"]) <= 0:
        raise ValueError("trainer_params.epochs 必须大于 0。")
    if int(trainer_params["batch_size"]) <= 0:
        raise ValueError("trainer_params.batch_size 必须大于 0。")
    if float(trainer_params["learning_rate"]) <= 0:
        raise ValueError("trainer_params.learning_rate 必须大于 0。")
    if float(trainer_params["grad_clip_norm"]) <= 0:
        raise ValueError("trainer_params.grad_clip_norm 必须大于 0。")
    if int(trainer_params["checkpoint_every"]) <= 0:
        raise ValueError("trainer_params.checkpoint_every 必须大于 0。")

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
    _validate_train_preflight(config_path, config, trainer_params)
    
    # 自动计算或校对 sqrt_embedding_dim
    if "sqrt_embedding_dim" not in model_params:
        model_params["sqrt_embedding_dim"] = math.sqrt(model_params["embedding_dim"])

    # 初始化 25 维模型
    model = Model(**model_params).to(device)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=trainer_params["learning_rate"])
    nar_pos_weight = torch.tensor([trainer_params["nar_pos_weight"]], device=device)
    nar_criterion = nn.BCEWithLogitsLoss(pos_weight=nar_pos_weight)
    ar_criterion = nn.CrossEntropyLoss(ignore_index=model.PAD_TOKEN, reduction="none")

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
        remapped_global_nar_labels = 0
        fixed_missing_depot_labels = 0
        
        for batch_idx, batch in enumerate(dataloader):
            optimizer.zero_grad()
            valid_items = 0
            losses = []

            for sample in batch:
                state_dict = sample["state_dict"]
                
                # 搬运基础标签
                global_node_indices = state_dict.get("global_node_indices", [])
                normalized_nar_labels, nar_label_status = _remap_nar_labels(
                    sample.get("nar_labels", []),
                    global_node_indices,
                    model.problem_size,
                )
                if nar_label_status == "global_to_local":
                    remapped_global_nar_labels += 1
                elif nar_label_status == "local_missing_depot":
                    fixed_missing_depot_labels += 1
                nar_labels = torch.tensor(
                    normalized_nar_labels, dtype=torch.float32, device=device
                ).unsqueeze(0)
                raw_ar_sequence = sample["ar_sequences"]
                mapped_ar_sequence = _remap_ar_sequence(
                    raw_ar_sequence,
                    global_node_indices,
                    model.problem_size,
                    model.PAD_TOKEN,
                )
                ar_sequences = torch.tensor(
                    mapped_ar_sequence, dtype=torch.long, device=device
                ).unsqueeze(0)

                # --- 组装 Mock 环境对象 ---
                mock_feat = MockProblemFeat()
                mock_feat.depot_xy = state_dict["depot_xy"].unsqueeze(0).to(device)
                mock_feat.node_xy = state_dict["node_xy"].unsqueeze(0).to(device)
                mock_feat.node_demand = state_dict["node_demand"].unsqueeze(0).to(device)

                reset_state = MockResetState()
                reset_state.problem_feat = mock_feat
                reset_state.tour_index = state_dict["tour_index"].unsqueeze(0).to(device)
                reset_state.neighbours = state_dict["neighbours"].unsqueeze(0).to(device)
                reset_state.tour_index = reset_state.tour_index.long().clamp(min=-1)
                local_num_customers = state_dict["node_xy"].shape[0]
                # Neighbour indices are in node space with depot included: valid range is [0, local_num_customers].
                reset_state.neighbours = reset_state.neighbours.long().clamp(
                    min=0, max=local_num_customers
                )
                
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
                flat_logits = ar_logits.reshape(-1, model.vocab_size)
                flat_target = ar_target.reshape(-1)
                token_losses = ar_criterion(flat_logits, flat_target).reshape_as(ar_target)

                # 对 AR 序列按删除/插入阶段加权（论文: wdelete / winsert）
                # 约定：target 第 0 位对应删除阶段，之后奇偶交替
                ar_positions = torch.arange(ar_target.shape[1], device=device).unsqueeze(0)
                token_weights = torch.where(
                    (ar_positions % 2) == 0,
                    torch.full_like(token_losses, trainer_params["ar_delete_weight"]),
                    torch.full_like(token_losses, trainer_params["ar_insert_weight"]),
                )
                valid_mask = (ar_target != model.PAD_TOKEN).float()
                weighted_losses = token_losses * token_weights * valid_mask
                denom = valid_mask.sum().clamp_min(1.0)
                loss_ar = weighted_losses.sum() / denom

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
        if remapped_global_nar_labels > 0:
            print(f"  ℹ️ 已重映射 {remapped_global_nar_labels} 条全局 NAR 标签到局部空间")
        if fixed_missing_depot_labels > 0:
            print(f"  ℹ️ 已修复 {fixed_missing_depot_labels} 条缺失 Depot 的 NAR 标签")

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
    try:
        train(args.config, seed=args.seed)
    except Exception as e:
        print(f"\n❌ 训练流程崩溃: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
