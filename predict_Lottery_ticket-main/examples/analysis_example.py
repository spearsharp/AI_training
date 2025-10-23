# -*- coding: utf-8 -*-
"""
Data analysis example

Demonstrates how to use data analysis features to analyze lottery data

Author: KittenCN
"""
import sys
import os

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.analysis import BasicAnalysis


def analysis_example():
    """Data analysis example"""
    print("=== 彩票Data analysis example ===")
    
    # Sample data
    sample_data = [
        [1, 5, 12, 23, 34, 45, 67, 78, 80, 77, 66, 55, 44, 33, 22, 11, 2, 3, 4, 6],
        [2, 8, 15, 29, 38, 47, 59, 68, 79, 71, 62, 53, 42, 31, 28, 17, 9, 7, 13, 19],
        [3, 11, 18, 27, 36, 49, 58, 69, 72, 64, 56, 48, 39, 21, 14, 26, 35, 41, 52, 63]
    ]
    
    print("Sample data分析结果:")
    try:
        datacnt, dataori = BasicAnalysis(sample_data)
        print(f"\nAnalysis complete, analyzed total of {len(sample_data)} data groups")
        print("出现频率统计已显示在上方")
    except Exception as e:
        print(f"分析失败: {e}")
    
    print("\n=== Example完成 ===")


if __name__ == "__main__":
    analysis_example()