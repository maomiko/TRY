import logging
import argparse
from pathlib import Path
from typing import List

import hydra
from omegaconf import DictConfig, OmegaConf

# 导入你写好的 Search 类
from src.search_sa import Search

def main(cfg: DictConfig) -> None:
    # 初始化日志
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("root")
    logger.info("开始生成 L2Seg 训练数据...")

    # 解析配置
    env_params = OmegaConf.to_container(cfg.env_params, resolve=True)
    tester_params = OmegaConf.to_container(cfg.tester_params, resolve=True)

    # 实例化并运行 Search
    searcher = Search(
        env_params=env_params,
        tester_params=tester_params,
    )

    # run() 方法会自动执行搜索、调用 label_generator，并保存 .pt 文件
    searcher.run()

def _parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate L2Seg Data with config file")
    parser.add_argument(
        "config", type=str, help="配置文件名，例如 test_cvrp_100.yaml"
    )
    parser.add_argument(
        "overrides", nargs="*", help="Override configuration parameters (key=value)"
    )
    return parser.parse_args(argv)

if __name__ == "__main__":
    args = _parse_args()

    # 假设你的配置文件放在 configs 目录下
    # 如果你的 yaml 直接放在根目录，这里需要适当调整路径
    config_file = Path("configs") / args.config
    config_dir = str(config_file.parent)
    config_name = config_file.name

    with hydra.initialize(config_path=config_dir, version_base=None):
        cfg = hydra.compose(config_name=config_name, overrides=args.overrides)

    main(cfg)