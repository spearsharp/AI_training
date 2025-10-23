#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to translate Chinese comments to English in Python files.
This script performs batch replacements of common Chinese phrases.
"""

import os
import re
from pathlib import Path

# Translation dictionary for common phrases
TRANSLATIONS = {
    # General
    "历史数据下载脚本": "Historical lottery data download script",
    "示例": "Example",
    "从 500.com 下载历史开奖数据": "Download historical lottery draw data from 500.com",
    "彩票类型代码，如 ssq / dlt / kl8，默认 ssq": "Lottery type code, such as ssq / dlt / kl8, default is ssq",
    "起始期号（包含），默认从最早可用期开始": "Starting issue number (inclusive), defaults to earliest available",
    "结束期号（包含），默认至最新期": "Ending issue number (inclusive), defaults to latest",
    "快乐8 是否使用出球顺序数据（仅 kl8 有效）": "Whether to use draw sequence data for KL8 (only valid for kl8)",
    "不支持的彩票类型": "Unsupported lottery type",
    "有效选项": "valid options",
    "数据下载完成": "Data download completed",
    
    # Data fetcher
    "数据抓取模块，负责从 500.com 拉取彩票历史数据并保存到本地": "Data fetching module for retrieving lottery historical data from 500.com and saving locally",
    "特点": "Features",
    "使用带重试的 requests.Session，满足网络安全要求": "Uses requests.Session with retry logic to meet network security requirements",
    "输出 Pandas DataFrame，供预处理与训练使用": "Outputs Pandas DataFrame for preprocessing and training",
    "针对快乐8（kl8）提供顺序版与常规版两种下载模式": "Provides both sequence and regular download modes for KL8 lottery",
    "描述一次下载操作的元信息": "Metadata for a download operation",
    "封装网络访问逻辑，提供带重试与域名校验的 GET 方法": "Encapsulates network access logic, providing GET method with retry and domain validation",
    "禁止访问域名": "Access to domain not allowed",
    "未找到开奖号码数据表格": "Draw number data table not found",
    "解析开奖号码失败，未获取到任何数据": "Failed to parse draw numbers, no data retrieved",
    "获取指定彩票的最新期号": "Get the latest issue number for a specified lottery",
    "最新期号": "Latest issue number",
    "下载历史数据并保存到": "Download historical data and save to",
    "下载": "Downloading",
    "历史数据": "historical data",
    "数据下载完成，共": "Data download completed, total",
    "期，保存至": "issues, saved to",
    "加载本地已下载的历史数据": "Load locally downloaded historical data",
    "未找到": "Not found",
    "历史数据，请先执行下载": "historical data, please download first",
    "缺失": "Missing",
    "字段，数据损坏或格式异常": "field, data is corrupted or format is abnormal",
    
    # Column names
    "期数": "Issue",
    "期号": "Issue",
    "红球": "Red",
    "蓝球": "Blue",
    
    # More detailed translations
    "公共接口封装": "Common interface wrapper",
    "为脚本层提供以下能力": "Provides the following capabilities for script layer",
    "查询最新期号": "Query latest issue number",
    "训练模型": "Train model",
    "预测下一期开奖": "Predict next draw",
    "下载指定彩票的历史数据": "Download historical data for specified lottery",
    "返回指定彩票的当前期号": "Return current issue number for specified lottery",
    "高层训练接口，封装": "High-level training interface, wraps",
    "开始训练": "Starting training",
    "模型": "model",
    "训练完成": "Training completed",
    "使用最新模型预测下一期号码": "Use latest model to predict next draw numbers",
    "预测结果": "Prediction result",
    "训练与预测流程封装": "Training and prediction pipeline wrapper",
    "暴露的核心函数": "Exposed core functions",
    "基于历史数据训练模型并写入本地": "Train model based on historical data and save locally",
    "从磁盘加载已训练模型": "Load trained models from disk",
    "使用最新窗口数据给出预测结果": "Generate prediction results using latest window data",
    "可用数据不足": "Insufficient data available",
    "窗口大小": "window size",
    "生成的样本数为": "generated samples count is",
    "请增加历史期数或减小窗口": "please increase historical issues or decrease window size",
    "无法导入 TensorFlow": "Unable to import TensorFlow",
    "请使用项目推荐的虚拟环境": "Please use the project's recommended virtual environment",
    "并安装": "and install",
    "预测结果包含负数": "Prediction result contains negative numbers",
    "可能是模型输出异常": "possibly model output anomaly",
    "历史数据不足": "Insufficient historical data",
    "无法获取": "unable to get",
    "条窗口序列": "window sequences",
    "训练指定彩票模型，并返回训练摘要": "Train model for specified lottery and return training summary",
    "样本": "samples",
    "验证集": "validation set",
    "批大小": "batch size",
    "轮数": "epochs",
    "模型已保存至": "Model saved to",
    "训练摘要已写入": "Training summary written to",
    "从磁盘加载训练好的模型": "Load trained models from disk",
    "未找到已训练的模型目录": "Trained model directory not found",
    "载入模型": "Loading model",
    "下未找到": "not found under",
    "模型文件": "model file",
    "使用最新模型预测下一期开奖号码": "Use latest model to predict next draw numbers",
    "最新窗口特征取": "Latest window features taken from",
    "中的原始数组最后": "last",
    "条": "entries from original array in",
    "球位数": "ball positions",
    "贪心去重采样：每次选概率最大的未被选过的数字": "Greedy deduplication sampling: select unselected number with highest probability each time",
    "将已选过的数字概率置为": "Set probability of already selected numbers to",
    "避免重复": "to avoid duplication",
    "彩票AI预测系统核心模块": "Lottery AI prediction system core module",
    "该模块包含了彩票AI预测系统的核心功能，包括": "This module contains core functionality of the lottery AI prediction system, including",
    "数据获取和处理": "Data acquisition and processing",
    "模型训练和预测": "Model training and prediction",
    "数据分析工具": "Data analysis tools",
    "配置管理": "Configuration management",
    "封装单个号码序列的特征、标签与偏移信息": "Encapsulates features, labels, and offset info for a single number sequence",
    "数据缺失列": "Data missing columns",
    "基于历史数据构建训练所需的": "Build training-required",
    "数组": "arrays based on historical data",
    "将窗口数据按比例划分训练与验证集合": "Split window data proportionally into training and validation sets",
    "必须介于": "must be between",
    "之间": "",
    "项目全局配置模块": "Project global configuration module",
    "该模块负责": "This module is responsible for",
    "读取": "Reading",
    "获取运行时配置": "to get runtime configuration",
    "定义彩票玩法的模型超参数与默认训练设置": "Define model hyperparameters and default training settings for lottery games",
    "提供路径常量与工具函数，供数据、模型与脚本复用": "Provide path constants and utility functions for reuse by data, models, and scripts",
    "升级": "Upgrade",
    "描述单个序列模型的结构参数": "Describes structural parameters of a single sequence model",
    "描述单种彩票玩法的训练所需配置": "Describes required configuration for training a single lottery game",
    "未找到系统配置文件": "System configuration file not found",
    "双色球": "Double Color Ball",
    "大乐透": "Super Lotto",
    "排列三": "P3",
    "七星彩": "7-Star Lottery",
    "快乐8": "Happy 8",
    "福彩3D": "3D Lottery",
    "确保项目运行所需的目录存在": "Ensure required directories for project execution exist",
    "根据玩法代码获取配置": "Get configuration by game code",
    "未知的彩票类型": "Unknown lottery type",
    "KL8相关功能已迁移至独立项目": "KL8-related features have been migrated to independent project",
    "请先执行下载": "please download first",
    "相关功能已迁移至独立项目": "related features have been migrated to independent project",
    "字段": "field",
    "数据损坏或格式异常": "data is corrupted or format is abnormal",
    "共": "total",
    
    # Fix mixed translations
    "查询Latest issue number": "Query latest issue number",
    "Downloading指定彩票的historical data": "Download historical data for specified lottery",
    "加载本地已Downloading的historical data": "Load locally downloaded historical data",
    "Not found {cfg.name} historical data，请先执行Downloading": "Historical data for {cfg.name} not found, please download first",
    "请先执行Downloading": "please download first",
    "Redmodel训练epochs": "Red model training epochs",
    "Bluemodel训练epochs": "Blue model training epochs", 
    "训练前自动Downloading最新数据": "Automatically download latest data before training",
    "开始Downloading数据": "Starting to download data",
    "基于historical dataTrain model并写入本地": "Train model based on historical data and save locally",
    "从磁盘加载已Train model": "Load trained model from disk",
    "使用最新window数据给出Prediction result": "Generate prediction result using latest window data",
    "window大小": "window size",
    "请增加历史Issue或减小window": "please increase historical issues or decrease window size",
    "历史Issue": "historical issues",
    "/conda 环境and install": " or conda environment and install",
    "Prediction result包含负数": "Prediction result contains negative numbers",
    "historical data不足": "Insufficient historical data",
    "entries from original array inwindow序列": "window sequences from original array",
    "训练指定彩票model": "Train model for specified lottery",
    "model已保存至": "Model saved to",
    "Not found已训练的model目录": "Trained model directory not found",
    "载入model": "Loading model",
    "下Not found red/blue model文件": ", red/blue model files not found",
    "使用最新modelPredict next draw号码": "Use latest model to predict next draw numbers",
    "最新window特征取 prepare_training_arrays last window entries from original array in": "Latest window features taken from last window entries in prepare_training_arrays",
    "将Prediction result转换回原始编号": "Convert prediction result back to original numbering",
    "，并针对Red/Blue输出": ", targeting Red/Blue outputs",
    "逐位置的类别概率": "position-wise category probabilities",
    "创建并激活项目推荐的 conda 环境 (environment.yml)": "Create and activate the project's recommended conda environment (environment.yml)",
    "安装 TensorFlow (Intel Mac / x86_64)": "Install TensorFlow (Intel Mac / x86_64)",
    "Apple Silicon (arm64) 用户可安装": "Apple Silicon (arm64) users can install",
    "或在激活环境后运行": "or run after activating environment",
    "TensorFlow 导入失败": "TensorFlow import failed",
    "请按下列步骤设置环境后重试": "Please set up the environment following these steps and retry",
    "将球位与时间维度交换，便于对每个球做 LSTM": "Swap ball position and time dimensions to facilitate LSTM for each ball",
    "便于对每个球做": "to facilitate",
    "构建指定彩票的红/Bluemodel": "Build red/blue models for specified lottery",
    "定义彩票玩法的model超参数与默认训练设置": "Define model hyperparameters and default training settings for lottery games",
    "供数据、model与脚本复用": "for reuse by data, models, and scripts",
    "描述单个序列model的结构参数": "Describes structural parameters of a single sequence model",
    "Not found系统配置文件": "System configuration file not found",
    "使用最新model预测下一Issue码": "Use latest model to predict next issue numbers",
    "返回指定彩票的当前Issue": "Return current issue for specified lottery",
    "（TensorFlow 2.15+ 版本）": "(TensorFlow 2.15+ version)",
    
    # analysis.py translations
    "输入要缩水至几个数? (-1表示结束)": "How many numbers to reduce to? (-1 to end)",
    "输入和值最小和最大值，用'，'分隔": "Enter min and max sum values, comma-separated",
    "输入要统计前几位和值:": "Enter how many top positions to calculate sum:",
    "Reading预测数据并分析": "Read prediction data and analyze",
    "缩水": "Reduce",
    "和值分析": "Sum value analysis",
    "退出": "Exit",
    "输入数据组数，-1为从文件输入:": "Enter number of data groups, -1 to read from file:",
    "输入第 #{} 组数据:": "Enter data group #{}:",
    "输入文件名:": "Enter filename:",
    "total有{}组数据，输入要分析前多少组：": "Total of {} data groups, enter how many groups to analyze:",
    "输入当前获奖数据，-1为结束：": "Enter current winning data, -1 to end:",
    "第{}组数据中，当前获奖数据出现的次数为{}次，概率为：{:.2f}%": "In data group {}, current winning data appears {} times, probability: {:.2f}%",
    "命中数 / 总预测数: {} / {}": "Hits / Total predictions: {} / {}",
    "输入要分析的数据组数，-1为全部:": "Enter number of data groups to analyze, -1 for all:",
    
    # examples translations
    "数据分析Example": "Data analysis example",
    "展示如何使用数据分析功能分析彩票数据": "Demonstrates how to use data analysis features to analyze lottery data",
    "添加src目录到Python路径": "Add src directory to Python path",
    "彩票数据分析Example": "Lottery data analysis example",
    "Example数据": "Sample data",
    "Example数据分析结果:": "Sample data analysis results:",
    "分析完成，total分析 {len(sample_data)} 组数据": "Analysis complete, analyzed total of {len(sample_data)} data groups",
    
    # Common phrases
    "模型训练脚本": "Model training script",
    "训练指定彩票的 LSTM 模型": "Train LSTM model for specified lottery",
    "时间窗口大小，默认使用玩法配置": "Time window size, defaults to game configuration",
    "训练批大小，默认使用玩法配置": "Training batch size, defaults to game configuration",
    "红球模型训练轮数": "Training epochs for red ball model",
    "蓝球模型训练轮数": "Training epochs for blue ball model",
    "训练前自动下载最新数据": "Automatically download latest data before training",
    "开始下载数据": "Starting data download",
    "训练完成，详情见": "Training completed, see details in",
    "模型构建模块": "Model building module",
    "提供基于": "Provides",
    "的多层 LSTM 序列模型": "multi-layer LSTM sequence model",
    "针对红球/蓝球输出逐位置的类别概率": "Outputs position-wise category probabilities for red/blue balls",
    "对每个球位独立应用 LSTM，提取窗口维度特征": "Apply LSTM independently to each ball position, extract window dimension features",
    "根据给定规格构建序列模型": "Build sequence model according to given specifications",
    "构建指定彩票的红/蓝球模型": "Build red/blue ball models for specified lottery",
    "构建模型": "Building model",
    "窗口": "window",
    "序列长": "sequence length",
    "类别数": "number of classes",
}

def translate_file(file_path: Path):
    """Translate Chinese comments in a single file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Apply translations
        for chinese, english in TRANSLATIONS.items():
            content = content.replace(chinese, english)
        
        # Only write if changes were made
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✓ Translated: {file_path}")
            return True
        return False
    except Exception as e:
        print(f"✗ Error processing {file_path}: {e}")
        return False

def main():
    """Main function to process all Python files."""
    project_root = Path(__file__).resolve().parents[1]
    python_files = list(project_root.rglob("*.py"))
    
    print(f"Found {len(python_files)} Python files")
    print("Starting translation...\n")
    
    translated_count = 0
    for py_file in python_files:
        # Skip this script itself and __pycache__
        if "__pycache__" in str(py_file) or py_file.name == "translate_comments.py":
            continue
        
        if translate_file(py_file):
            translated_count += 1
    
    print(f"\n✓ Translation complete: {translated_count} files modified")

if __name__ == "__main__":
    main()
