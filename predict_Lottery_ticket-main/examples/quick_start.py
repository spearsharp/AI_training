# -*- coding: utf-8 -*-
"""
彩票预测系统快速开始Example

本Example展示如何使用彩票预测系统进行数据获取、model训练和预测

Author: KittenCN
"""
import sys
import os

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.common import get_current_number, get_data_run
from src.config import LOTTERY_CONFIGS


def quick_start_example():
    """快速开始Example"""
    print("=== 彩票AI预测系统快速开始Example ===")
    
    # 1. 获取数据
    print("\n1. 获取Double Color Ball数据...")
    try:
        get_data_run('ssq')
        print("数据获取成功！")
    except Exception as e:
        print(f"数据获取失败: {e}")
        return
    
    # 2. 查看当前Issue
    print("\n2. 查看当前Issue...")
    try:
        current_num = get_current_number('ssq')
        print(f"Double Color BallLatest issue number: {current_num}")
    except Exception as e:
        print(f"获取Issue失败: {e}")
    
    # 3. 显示支持的彩票类型
    print("\n3. 支持的彩票类型:")
    for code, cfg in LOTTERY_CONFIGS.items():
        print(f"  - {code}: {cfg.name}")
    
    print("\n=== Example完成 ===")
    print("\n下一步:")
    print("1. 使用 'python scripts/train.py --name ssq --window-size 5 --red-epochs 60' Train model")
    print("2. 使用 'python scripts/predict.py --name ssq --window-size 5 --save' 进行预测")


if __name__ == "__main__":
    quick_start_example()
