"""Training loop and utilities for model optimization."""

import os
import time
import warnings
from logging import getLogger
from typing import Tuple, Dict, Any
from functools import partial

from torch.nn.utils.rnn import pad_sequence

import numpy as np
import torch
from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler
import torch
from torch.utils.data import Dataset, DataLoader


from .model import Model
from .env import Env
from .logging_utils import (
    get_result_folder,
    TimeEstimator,
    AverageMeter,
)

from .seed_sampler import SeedVectorSampler
import wandb

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# --- 新增：用于伪装环境对象的类 ---
class DummyProblemFeat:
    pass

class DummyResetState:
    pass
# ---------------------------------


def l2s_collate_fn(batch, global_end_token: int, pad_token: int):
    """
    处理变长切片数据：
    1. 将全局节点 ID 映射为局部 ID。
    2. 对所有的变长特征 (node_xy, nar_labels 等) 进行 Padding 补齐。
    3. 生成 pad_mask，告诉模型哪些是假节点。
    """
    nar_labels_list = []
    ar_seqs_list = []
    node_xy_list = []
    node_demand_list = []
    tour_index_list = []
    

    neighbours_list = []

    node_tw_list = []
    node_prizes_list = []
    has_tw = 'node_tw' in batch[0]['state_dict']
    has_prizes = 'node_prizes' in batch[0]['state_dict']


    unknown_ar_tokens = set()

    for item in batch:
        global_indices = item['state_dict']['global_node_indices']

        # 在前向传播时，Depot 永远会被 concat 在最前面（索引 0）
        # 所以客户的局部索引必须从 1 开始顺延
        g2l_map = {g_id: l_idx + 1 for l_idx, g_id in enumerate(global_indices)}
        g2l_map[0] = 0  # 手动补上 Depot 的映射
            
        local_ar_seq = []
        for raw_g_id in item['ar_sequences']:
            g_id = int(raw_g_id)
            if g_id in g2l_map:
                local_ar_seq.append(g2l_map[g_id])
            elif g_id == int(global_end_token):
                # 合法结束符，保留全局固定索引
                local_ar_seq.append(int(global_end_token))
            else:
                # 异常 token 回退为 END，同时显式告警，避免静默吞错
                unknown_ar_tokens.add(g_id)
                local_ar_seq.append(int(global_end_token))
                    
        ar_seqs_list.append(torch.tensor(local_ar_seq, dtype=torch.long))
        
        # 将其他特征转为 Tensor
        nar_labels_list.append(torch.tensor(item['nar_labels'], dtype=torch.float32))
        node_xy_list.append(item['state_dict']['node_xy'])
        node_demand_list.append(item['state_dict']['node_demand'])
        tour_index_list.append(item['state_dict']['tour_index'])
        neighbours_list.append(item['state_dict']['neighbours'])  
    
        # 【新增】：抽取拓展特征
        if has_tw:
            node_tw_list.append(item['state_dict']['node_tw'])
        if has_prizes:
            node_prizes_list.append(item['state_dict']['node_prizes'])
    
    if unknown_ar_tokens:
        preview = sorted(unknown_ar_tokens)[:8]
        warnings.warn(
            (
                "Unknown AR token IDs detected in batch and remapped to END token. "
                f"count={len(unknown_ar_tokens)}, preview={preview}, end_token={int(global_end_token)}"
            ),
            RuntimeWarning,
            stacklevel=2,
        )

    # ==========================================
    # 2. 动态 Padding (补齐长短不一的 Tensor)
    # ==========================================
    # 找出当前 Batch 里节点最多的一张切片图（仅用于特征 padding）
    max_nodes = max(len(x) for x in nar_labels_list)
    ar_sequences_padded = pad_sequence(
        ar_seqs_list, batch_first=True, padding_value=int(pad_token)
    )
    nar_labels_padded = pad_sequence(nar_labels_list, batch_first=True, padding_value=0.0)
    node_xy_padded = pad_sequence(node_xy_list, batch_first=True, padding_value=0.0)
    node_demand_padded = pad_sequence(node_demand_list, batch_first=True, padding_value=0.0)
    tour_index_padded = pad_sequence(tour_index_list, batch_first=True, padding_value=-1)
    
    # ==========================================
    # 3. 生成注意力掩码 (Attention Mask)
    # ==========================================
    # True 表示是 Pad 出来的假节点，False 表示是真实节点
    pad_mask = torch.zeros((len(batch), max_nodes), dtype=torch.bool)
    for i, seq in enumerate(nar_labels_list):
        pad_mask[i, len(seq):] = True

    # 4. 还原 ResetState 对象结构
    reset_state = DummyResetState()
    reset_state.problem_feat = DummyProblemFeat()
    
    # Depot 车场坐标只有一个，永远是 (batch, 2)，可以直接 stack
    reset_state.problem_feat.depot_xy = torch.stack([item['state_dict']['depot_xy'] for item in batch])
    
    # 其他全部用 Padded 后的张量
    reset_state.problem_feat.node_xy = node_xy_padded
    reset_state.problem_feat.node_demand = node_demand_padded
    reset_state.tour_index = tour_index_padded
    
    # 新增属性：把 pad_mask 传下去，一会 model.py 里的 Encoder 会用到它！
    reset_state.pad_mask = pad_mask 

    # 用 0 (Depot) 补齐假节点的邻居
    neighbours_padded = pad_sequence(neighbours_list, batch_first=True, padding_value=0)
    
    # 赋值给 reset_state
    reset_state.neighbours = neighbours_padded

    if has_tw:
        reset_state.problem_feat.node_tw = pad_sequence(node_tw_list, batch_first=True, padding_value=0.0)
    if has_prizes:
        reset_state.problem_feat.node_prizes = pad_sequence(node_prizes_list, batch_first=True, padding_value=0.0)

    # ==========================================
    # [核心植入]：在训练 DataLoader 中实时计算原版特征
    # ==========================================
    from .utils import compute_original_l2seg_features
    static_feats, dynamic_feats = compute_original_l2seg_features(
        reset_state.problem_feat.depot_xy,
        reset_state.problem_feat.node_xy,
        reset_state.problem_feat.node_demand,
        reset_state.tour_index,
        reset_state.neighbours,
        pad_mask=reset_state.pad_mask
    )
    
    reset_state.l2seg_static_feats = static_feats
    reset_state.l2seg_dynamic_feats = dynamic_feats
    # ==========================================



    return {
        'nar_labels': nar_labels_padded,
        'ar_sequences': ar_sequences_padded,
        'reset_state': reset_state,
        'pad_token': int(pad_token)  # 传给 Loss 函数，用于忽略 Padding 部分
    }

class L2SDataset(Dataset):
    def __init__(self, data_path):
        self.data = torch.load(data_path, map_location="cpu", weights_only=True)
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        # 直接返回一整条记录字典
        return self.data[idx]


class Trainer:
    """Manages the full training lifecycle: setup, training loop, logging, and checkpoints."""

    def __init__(
        self,
        env_params: Dict[str, Any],
        model_params: Dict[str, Any],
        optimizer_params: Dict[str, Any],
        trainer_params: Dict[str, Any],
        logger_params: Dict[str, Any],        
    ):
        """Initialize trainer with configuration parameters."""
        # Store configuration
        self.env_params = env_params
        self.model_params = model_params
        self.optimizer_params = optimizer_params
        self.trainer_params = trainer_params

        # Setup logging and directories
        self.logger = getLogger(name="trainer")
        self.results_dir = get_result_folder()
        self.time_estimator = TimeEstimator()

        # Setup device
        self.device = self._setup_device()

        # Initialize core components
        self.model = Model(**self.model_params).to(self.device)
        self.model_frozen = Model(**self.model_params).to(self.device)
        self.env = Env(**self.env_params)
        self.optimizer = Optimizer(
            self.model.parameters(), **self.optimizer_params["optimizer"]
        )
        self.scheduler = Scheduler(self.optimizer, **self.optimizer_params["scheduler"])
        # Use new AMP GradScaler API; enable scaling only on CUDA
        if self.device.type == "cuda":
            self.scaler = torch.amp.GradScaler("cuda")
        else:
            self.scaler = torch.amp.GradScaler(enabled=False)

        # Training parameters
        self.batch_size = self.trainer_params["train_batch_size"]
        self.rollout_size = self.trainer_params["rollout_size"]

        # Seed vector sampler
        self.seed_sampler = SeedVectorSampler(model_params["z_dim"], self.device)

        # Restore from checkpoint if needed
        self.start_epoch = 1
        self.wandb_run_id = None
        self._load_checkpoint_if_exists()



        # Setup experiment tracking
        self.use_wandb = logger_params["wandb"]["enable"]
        if self.use_wandb:
            self._init_wandb(
                logger_params,
                env_params,
                model_params,
                optimizer_params,
                trainer_params,
            )
        l2s_data_path = self.trainer_params.get("l2s_data_path", "results/l2seg_dataset/l2seg_training_data.pt")
        if os.path.exists(l2s_data_path):
            full_dataset = L2SDataset(l2s_data_path)
            
            # 【新增】：按 9:1 随机切分训练集和验证集
            dataset_size = len(full_dataset)
            val_size = int(0.1 * dataset_size)
            train_size = dataset_size - val_size
            
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                full_dataset, [train_size, val_size]
            )
            
            # 创建训练集 DataLoader
            collate = partial(
                l2s_collate_fn,
                global_end_token=int(self.model.vocab_size - 1),
                pad_token=int(self.model.PAD_TOKEN),
            )
            self.train_dataloader = DataLoader(
                self.train_dataset, 
                batch_size=self.batch_size, 
                shuffle=True, 
                num_workers=4,
                collate_fn=collate  
            )
            # 创建验证集 DataLoader (不需要打乱)
            self.val_dataloader = DataLoader(
                self.val_dataset, 
                batch_size=self.batch_size, 
                shuffle=False, 
                num_workers=4,
                collate_fn=collate  
            )
            self.logger.info(f"Loaded L2Seg dataset: {train_size} train, {val_size} val.")
        else:
            self.logger.warning(f"L2Seg data not found at {l2s_data_path}. Please generate data first.")
            self.train_dataloader = []
            self.val_dataloader = []

    def _setup_device(self) -> torch.device:
        """Setup and return the compute device (CPU or CUDA)."""
        use_cuda = self.trainer_params["use_cuda"]
        if use_cuda:
            cuda_device_num = self.trainer_params["cuda_device_num"]
            torch.cuda.set_device(cuda_device_num)
            return torch.device("cuda", cuda_device_num)
        return torch.device("cpu")

    def _load_checkpoint_if_exists(self) -> None:
        """Load model from explicit checkpoint or auto-resume from latest."""
        # Load from explicit checkpoint if specified
        model_load = self.trainer_params["model_load"]
        if model_load["enable"]:
            checkpoint_path = "{path}/checkpoint-{epoch}.pt".format(**model_load)
            checkpoint = torch.load(
                checkpoint_path, map_location=self.device, weights_only=False
            )
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.logger.info(f"Loaded model from {checkpoint_path}")

        # Auto-resume from latest if exists
        latest_path = os.path.join(self.results_dir, "latest_model.pt")
        if os.path.isfile(latest_path):
            checkpoint = torch.load(
                latest_path, map_location=self.device, weights_only=False
            )
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.scheduler.last_epoch = checkpoint["epoch"] - 1
            self.start_epoch = 1 + checkpoint["epoch"]
            self.wandb_run_id = checkpoint.get("wandb_run_id")
            self.logger.info(f"Resuming from epoch {self.start_epoch}")

    def _init_wandb(
        self,
        logger_params: Dict[str, Any],
        env_params: Dict[str, Any],
        model_params: Dict[str, Any],
        optimizer_params: Dict[str, Any],
        trainer_params: Dict[str, Any],
    ) -> None:
        """Initialize Weights & Biases experiment tracking."""
        run = wandb.init(
            project=logger_params["wandb"]["project"],
            name=logger_params["desc"],
            config={
                "env_params": env_params,
                "model_params": model_params,
                "optimizer_params": optimizer_params,
                "trainer_params": trainer_params,
            },
            id=self.wandb_run_id,
            resume="allow",
        )
        self.wandb_run_id = run.id

    def run(self) -> None:
        """Execute the main training loop across all epochs."""
        self.time_estimator.reset(self.start_epoch)
        total_epochs = self.trainer_params["epochs"]

        for epoch in range(self.start_epoch, total_epochs + 1):
            self.logger.info("=" * 80)

            # Train one epoch
            metrics = self._train_one_epoch(epoch)

            # Update learning rate
            self.scheduler.step()

            # Log timing and save checkpoints
            self._log_timing(epoch, total_epochs)
            self._save_checkpoints(epoch, total_epochs)

            # Run validation periodically
            if epoch % 5 == 0:
                self._validate_one_epoch(epoch)

            # Final announcement
            if epoch == total_epochs:
                self.logger.info("=" * 80)
                self.logger.info("Training Complete")
                self.logger.info("=" * 80)

    def _train_one_epoch(self, epoch: int) -> Tuple[float, float, float, float]:
        """
        使用 L2Seg 的监督学习方式训练一个 Epoch。
        返回 (nar_accuracy, total_loss, nar_loss, ar_loss)。
        """
        # 获取梯度累加步数，默认为1
        grad_acc_iterations = self.trainer_params.get("grad_acc_iterations", 1)

        # 初始化 L2Seg 专用的指标追踪器
        metrics = {
            "nar_accuracy": AverageMeter(),
            "loss": AverageMeter(),
            "loss_nar": AverageMeter(),
            "loss_ar": AverageMeter(),
        }
        
        processed_batches = 0
        logged_batches = 0
        epoch_start_time = time.time()
        
        # 获取 DataLoader 的总长度
        total_batches = len(self.train_dataloader)

        self.model.zero_grad()
        
        # 遍历 DataLoader 提供的监督学习数据集
        for batch_idx, batch in enumerate(self.train_dataloader):
            # 调用我们刚重写的 _train_one_batch
            nar_acc, loss, loss_nar, loss_ar = self._train_one_batch(batch)
            
            # 更新指标 (根据真实 batch_size 更新权重)
            batch_size = batch["nar_labels"].size(0)
            metrics["nar_accuracy"].update(nar_acc, batch_size)
            metrics["loss"].update(loss, batch_size)
            metrics["loss_nar"].update(loss_nar, batch_size)
            metrics["loss_ar"].update(loss_ar, batch_size)

            # 梯度累加与优化器步进
            if (batch_idx + 1) % grad_acc_iterations == 0 or (batch_idx + 1) == total_batches:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.model.zero_grad()

            processed_batches += 1

            # 在第一个 Epoch 打印前 10 个 batch 的详细进度
            if epoch == self.start_epoch and logged_batches < 10:
                self._log_batch_progress(
                    epoch, processed_batches, total_batches, metrics
                )
                logged_batches += 1

        # 打印 Epoch 总结
        self._log_epoch_summary(
            epoch, processed_batches, total_batches, metrics
        )

        # W&B 日志记录
        if self.use_wandb:
            self._log_to_wandb(
                epoch, metrics, time.time() - epoch_start_time
            )

        return (
            metrics["nar_accuracy"].avg,
            metrics["loss"].avg,
            metrics["loss_nar"].avg,
            metrics["loss_ar"].avg,
        )

    def _train_one_batch(self, batch: Dict[str, Any]) -> Tuple[float, float, float, float]:
        """
        使用监督学习 (L2Seg) 训练一个 Batch。
        返回: (nar_准确率, 总损失, nar损失, ar损失) - 用于替换原本的 (score, loss, reward, nb_improved)
        """
        self.model.train()

        # 1. 从 DataLoader 解包数据并转移到设备
        # 注意：你需要将环境状态打包成 reset_state 字典或对象传入
        reset_state = batch["reset_state"] 
        nar_labels_target = batch["nar_labels"].to(self.device)  # shape: (batch_size, problem_size + 1)
        ar_sequences = batch["ar_sequences"].to(self.device)    # shape: (batch_size, max_seq_len)
        pad_token = batch["pad_token"]  # <--- 新增这行，接住传过来的 pad_token

        # 2. 编码器前向传播 (提取图特征)
        # 监督学习通常不需要 seed vector (z) 来做探索，可以传 None 或者特定的 context
        with torch.amp.autocast(device_type=self.device.type):
            self.model.pre_forward(reset_state, z=None)

            # 3. 双头预测 (需要在 model.py 中新增这两个方法)
            # NAR预测：每个节点是否需要被破坏/保留？
            nar_logits = self.model.nar_forward() 
            
            # AR预测：自回归生成序列 (Teacher Forcing，将真实序列作为输入引导预测)
            ar_logits = self.model.ar_forward(ar_sequences)

            # 4. 计算 L2Seg 的多任务损失
            loss, loss_nar, loss_ar = self._compute_l2seg_loss(
                nar_logits, nar_labels_target, 
                ar_logits, ar_sequences,
                pad_token  # <--- 新增传参
            )

        # 5. 反向传播与梯度缩放
        grad_acc_iterations = max(1, int(self.trainer_params.get("grad_acc_iterations", 1)))
        self.scaler.scale(loss / grad_acc_iterations).backward()

        # 6. 计算 NAR 准确率作为评估指标 (代替原本的 RL Reward)
        with torch.no_grad():
            # 使用 Sigmoid 将 logits 转为概率，> 0.5 视为预测分类 1
            nar_preds = (torch.sigmoid(nar_logits) > 0.5).float()
            nar_accuracy = (nar_preds == nar_labels_target).float().mean()

        return nar_accuracy.item(), loss.item(), loss_nar.item(), loss_ar.item()
    
    
    def _validate_one_epoch(self, epoch: int) -> None:
        """
        极速监督学习验证：在保留的验证集上计算 Loss 和准确率。
        不运行真实的 VRP 搜索，耗时仅需几秒。
        """
        self.logger.info("-" * 80)
        self.logger.info(f"Running Validation for Epoch {epoch}...")
        
        # 开启评估模式
        self.model.eval()
        
        metrics = {
            "nar_accuracy": AverageMeter(),
            "loss": AverageMeter(),
            "loss_nar": AverageMeter(),
            "loss_ar": AverageMeter(),
        }
        
        with torch.no_grad():
            for batch in self.val_dataloader:
                reset_state = batch["reset_state"] 
                nar_labels_target = batch["nar_labels"].to(self.device)  
                ar_sequences = batch["ar_sequences"].to(self.device)    
                pad_token = batch["pad_token"]
                
                with torch.amp.autocast(device_type=self.device.type):
                    self.model.pre_forward(reset_state, z=None)
                    nar_logits = self.model.nar_forward() 
                    ar_logits = self.model.ar_forward(ar_sequences)
                    
                    loss, loss_nar, loss_ar = self._compute_l2seg_loss(
                        nar_logits, nar_labels_target, ar_logits, ar_sequences, pad_token
                    )
                    
                # 计算 NAR 准确率
                nar_preds = (torch.sigmoid(nar_logits) > 0.5).float()
                nar_accuracy = (nar_preds == nar_labels_target).float().mean()
                
                # 更新指标
                batch_size = nar_labels_target.size(0)
                metrics["nar_accuracy"].update(nar_accuracy.item(), batch_size)
                metrics["loss"].update(loss.item(), batch_size)
                metrics["loss_nar"].update(loss_nar.item(), batch_size)
                metrics["loss_ar"].update(loss_ar.item(), batch_size)
                
        # 打印验证总结
        self.logger.info(
            f"Validation Epoch {epoch} Summary |  "
            f'Total Loss: {metrics["loss"].avg:6.4f}  |  '
            f'NAR Acc: {metrics["nar_accuracy"].avg * 100:.2f}%'
        )
        self.logger.info("-" * 80)
        
        # 将验证指标同步到 Weights & Biases
        if self.use_wandb:
            import wandb
            wandb.log(
                step=epoch,
                data={
                    "val/total_loss": metrics["loss"].avg,
                    "val/loss_nar": metrics["loss_nar"].avg,
                    "val/loss_ar": metrics["loss_ar"].avg,
                    "val/nar_accuracy": metrics["nar_accuracy"].avg,
                },
            )


    
    def _compute_l2seg_loss(
        self, 
        nar_logits: torch.Tensor, 
        nar_labels: torch.Tensor, 
        ar_logits: torch.Tensor, 
        ar_sequences: torch.Tensor,
        pad_token: int  # <--- 新增参数
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        计算 L2Seg 的双头损失：
        - NAR: 二元交叉熵 (Binary Cross Entropy)
        - AR: 多分类交叉熵 (Cross Entropy)
        """
        import torch.nn.functional as F

        # ==========================================
        # 1. NAR 分支损失 (Non-Autoregressive)
        # ==========================================
        # 直接使用带有 Logits 的 BCE，数值计算更稳定
        loss_nar = F.binary_cross_entropy_with_logits(nar_logits, nar_labels.float())

        # ==========================================
        # 2. AR 分支损失 (Autoregressive)
        # ==========================================
        if ar_logits is None or ar_sequences.size(1) <= 1:
            loss_ar = torch.tensor(0.0, device=self.device)
        else:
            ar_targets = ar_sequences[:, 1:].clone()
            ar_logits = ar_logits[:, :-1, :]

            vocab_size = ar_logits.size(-1)
            ar_logits_flat = ar_logits.reshape(-1, vocab_size)
            ar_targets_flat = ar_targets.reshape(-1)

            token_losses = F.cross_entropy(
                ar_logits_flat,
                ar_targets_flat,
                ignore_index=pad_token,
                reduction="none",
            ).reshape_as(ar_targets)

            # 约定 AR 监督序列按 delete/insert 位置交替排列（与 train.py 保持一致）
            ar_positions = torch.arange(
                ar_targets.shape[1], device=self.device
            ).unsqueeze(0)
            delete_weight = float(self.trainer_params.get("ar_delete_weight", 1.0))
            insert_weight = float(self.trainer_params.get("ar_insert_weight", 1.0))
            token_weights = torch.where(
                (ar_positions % 2) == 0,
                delete_weight,
                insert_weight,
            )

            valid_mask = (ar_targets != pad_token).float()
            weighted_losses = token_losses * token_weights * valid_mask
            loss_ar = weighted_losses.sum() / valid_mask.sum().clamp_min(1.0)
            
        # ==========================================
        # 3. 总损失合并
        # ==========================================
        # 使用超参数控制 AR 损失的权重，通常在 trainer_params 中定义
        alpha = self.trainer_params.get("ar_loss_weight", 1.0)
        total_loss = loss_nar + alpha * loss_ar

        return total_loss, loss_nar, loss_ar


    def _log_batch_progress(
        self,
        epoch: int,
        processed: int,
        total: int,
        metrics: Dict[str, AverageMeter]
    ) -> None:
        """Log progress for a single batch."""
        self.logger.info(
            f"Epoch {epoch:3d}  |  Train {processed:4d}/{total:4d} ({100.0 * processed / total:5.1f}%)  |  "
            f'Total Loss: {metrics["loss"].avg:6.4f}  |  '
            f'NAR Loss: {metrics["loss_nar"].avg:6.4f}  |  AR Loss: {metrics["loss_ar"].avg:6.4f}  |  '
            f'NAR Acc: {metrics["nar_accuracy"].avg * 100:.2f}%'
        )

    def _log_epoch_summary(
        self,
        epoch: int,
        processed: int,
        total: int,
        metrics: Dict[str, AverageMeter]
    ) -> None:
        """Log summary for entire epoch."""
        self.logger.info(
            f"Epoch {epoch:3d} Summary |  "
            f'Total Loss: {metrics["loss"].avg:6.4f}  |  '
            f'NAR Loss: {metrics["loss_nar"].avg:6.4f}  |  AR Loss: {metrics["loss_ar"].avg:6.4f}  |  '
            f'NAR Acc: {metrics["nar_accuracy"].avg * 100:.2f}%'
        )

    def _log_to_wandb(
        self,
        epoch: int,
        metrics: Dict[str, AverageMeter],
        duration: float,
    ) -> None:
        """Log metrics to Weights & Biases."""
        import wandb
        wandb.log(
            step=epoch,
            data={
                "train/total_loss": metrics["loss"].avg,
                "train/loss_nar": metrics["loss_nar"].avg,
                "train/loss_ar": metrics["loss_ar"].avg,
                "train/nar_accuracy": metrics["nar_accuracy"].avg,
                "time/epoch": duration,
            },
        )

    def _log_timing(self, epoch: int, total_epochs: int) -> None:
        """Log elapsed and remaining time estimates."""
        elapsed, remaining = self.time_estimator.get_est_string(epoch, total_epochs)
        self.logger.info(
            f"Epoch {epoch:3d}/{total_epochs:3d}  |  Elapsed: {elapsed}  |  Remain: {remaining}"
        )

    def _save_checkpoints(self, epoch: int, total_epochs: int) -> None:
        """Save model checkpoints (periodic and latest)."""
        model_save_interval = self.trainer_params["model_save_interval"]
        all_done = epoch == total_epochs

        # Save periodic checkpoint
        if all_done or (epoch % model_save_interval) == 0:
            self.logger.info("Saving checkpoint")
            checkpoint = self._build_checkpoint(epoch)
            torch.save(
                checkpoint, os.path.join(self.results_dir, f"checkpoint-{epoch}.pt")
            )

        # Always save latest
        checkpoint = self._build_checkpoint(epoch)
        torch.save(checkpoint, os.path.join(self.results_dir, "latest_model.pt"))



    def _build_checkpoint(self, epoch: int) -> Dict[str, Any]:
        """Build checkpoint dictionary."""
        return {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "model_params": self.model_params,
            "env_params": self.env_params,
            "wandb_run_id": self.wandb_run_id,
        }
