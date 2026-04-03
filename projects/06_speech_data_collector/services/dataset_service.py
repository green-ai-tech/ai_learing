"""
数据集管理服务
"""
import os
import zipfile
from typing import Tuple

import numpy as np


class DatasetManager:
    """数据集管理器，负责生成和导出数据集"""

    @staticmethod
    def update_dataset(root_dir: str) -> Tuple[bool, str]:
        """更新数据集，生成testing_list.txt和validation_list.txt"""
        # 获取所有音频文件
        audio_files = []
        for label in ['向上', '向下']:
            label_dir = os.path.join(root_dir, label)
            if os.path.exists(label_dir):
                for file in os.listdir(label_dir):
                    if file.endswith('.wav'):
                        rel_path = os.path.join(label, file)
                        audio_files.append(rel_path)

        if not audio_files:
            return False, "没有找到音频文件"

        # 分割训练集和验证集（80%训练，20%验证）
        np.random.seed(42)  # 固定随机种子
        indices = np.random.permutation(len(audio_files))
        split_idx = int(len(audio_files) * 0.8)

        train_files = [audio_files[i] for i in indices[:split_idx]]
        val_files = [audio_files[i] for i in indices[split_idx:]]

        # 写入validation_list.txt
        val_path = os.path.join(root_dir, 'validation_list.txt')
        with open(val_path, 'w', encoding='utf-8') as f:
            for file in val_files:
                f.write(file + '\n')

        # 写入testing_list.txt（这里用剩余的训练集作为测试集，实际项目中可调整）
        test_path = os.path.join(root_dir, 'testing_list.txt')
        with open(test_path, 'w', encoding='utf-8') as f:
            for file in train_files:
                f.write(file + '\n')

        return True, f"数据集已更新！\n训练集: {len(train_files)}个\n验证集: {len(val_files)}个"

    @staticmethod
    def export_dataset(root_dir: str, export_path: str) -> Tuple[bool, str]:
        """导出数据集为ZIP文件"""
        try:
            with zipfile.ZipFile(export_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, dirs, files in os.walk(root_dir):
                    for file in files:
                        if file.endswith('.wav') or file.endswith('.txt'):
                            file_path = os.path.join(root, file)
                            arcname = os.path.relpath(file_path, root_dir)
                            zipf.write(file_path, arcname)
            return True, f"数据集已导出到:\n{export_path}"
        except Exception as e:
            return False, f"导出失败: {str(e)}"
