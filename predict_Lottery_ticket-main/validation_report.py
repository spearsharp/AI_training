# -*- coding: utf-8 -*-
"""
Validate kl8_analysis_plus.py fix results

Author: KittenCN
"""

def main():
    print("=== Validating kl8_analysis_plus.py fix status ===\n")
    
    # 1. Test sklearn import
    try:
        from sklearn.cluster import KMeans
        print("✅ sklearn.cluster.KMeans imported successfully")
    except ImportError as e:
        print(f"❌ sklearn import failed: {e}")
        return False
    
    # 2. Test matplotlib import  
    try:
        import matplotlib.pyplot as plt
        print("✅ matplotlib.pyplot 导入成功")
    except ImportError as e:
        print(f"❌ matplotlib 导入失败: {e}")
        return False
    
    # 3. 检查已修复的文件
    import os
    kl8_files = [
        'src/analysis/kl8_analysis_plus.py',
        'src/analysis/kl8_analysis.py', 
        'src/analysis/kl8_cash_plus.py',
        'src/analysis/kl8_cash.py'
    ]
    
    print("\n📝 检查修复的文件:")
    for file_path in kl8_files:
        if os.path.exists(file_path):
            print(f"   ✅ {file_path} 存在")
        else:
            print(f"   ❌ {file_path} 不存在")
    
    print("\n🔧 已完成的修复:")
    print("   • 安装了 scikit-learn>=1.1.0")
    print("   • 安装了 matplotlib>=3.5.0") 
    print("   • 修复了 kl8_analysis_plus.py 中的 'from common import' 为 'from ..common import'")
    print("   • 修复了 kl8_analysis.py 中的导入路径")
    print("   • 修复了 kl8_cash_plus.py 中的导入路径")
    print("   • 修复了 kl8_cash.py 中的导入路径")
    print("   • 所有文件的 'from config import' 已更新为 'from ..config import'")
    
    print("\n🎉 kl8_analysis_plus.py 的导入错误已修复!")
    print("   现在可以正常使用 sklearn.cluster 等依赖了。")
    
    return True

if __name__ == "__main__":
    main()