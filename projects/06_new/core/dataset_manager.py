#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集管理器 - 负责标签加载、文件树更新、数据集划分、导出
"""

import os
import shutil
import zipfile
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np

from config import (
    AUDIO_FILE_EXTENSION,
    DEFAULT_LABELS,
    EXPORT_FILENAME_PREFIX,
    RANDOM_SEED,
    TEXT_FILE_EXTENSION,
    TRAIN_RATIO,
)


class DatasetManager:
    """数据集管理器，处理标签、文件扫描、数据集划分和导出"""

    def __init__(self, base_dir: str) -> None:
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    def load_existing_labels(self) -> List[str]:
        """扫描基目录下已有的标签文件夹"""
        labels: List[str] = []
        if os.path.exists(self.base_dir):
            for item in os.listdir(self.base_dir):
                item_path = os.path.join(self.base_dir, item)
                if os.path.isdir(item_path) and not item.startswith("."):
                    labels.append(item)
        return labels

    def get_all_labels(self) -> List[str]:
        """合并默认标签和已有标签，去重"""
        existing = self.load_existing_labels()
        all_labels = list(set(DEFAULT_LABELS + existing))
        return all_labels

    def ensure_label_dirs(self, labels: List[str]) -> None:
        """确保所有标签目录存在"""
        for label in labels:
            os.makedirs(os.path.join(self.base_dir, label), exist_ok=True)

    def add_label(self, label: str) -> bool:
        """添加新标签并创建对应目录"""
        label_dir = os.path.join(self.base_dir, label)
        if os.path.exists(label_dir):
            return False
        os.makedirs(label_dir, exist_ok=True)
        return True

    def remove_label(self, label: str) -> bool:
        """删除标签及其目录"""
        label_dir = os.path.join(self.base_dir, label)
        if os.path.exists(label_dir):
            shutil.rmtree(label_dir)
            return True
        return False

    def get_audio_files(self) -> List[str]:
        """获取所有音频文件的相对路径列表"""
        audio_files: List[str] = []
        labels = self.get_all_labels()
        for label in labels:
            label_dir = os.path.join(self.base_dir, label)
            if os.path.exists(label_dir):
                for file in os.listdir(label_dir):
                    if file.endswith(AUDIO_FILE_EXTENSION):
                        rel_path = os.path.join(label, file)
                        audio_files.append(rel_path)
        return audio_files

    def split_dataset(
        self,
    ) -> Tuple[List[str], List[str]]:
        """
        将数据集划分为训练集和验证集。
        返回 (train_files, val_files)
        """
        audio_files = self.get_audio_files()
        if not audio_files:
            return [], []

        np.random.seed(RANDOM_SEED)
        indices = np.random.permutation(len(audio_files))
        split_idx = int(len(audio_files) * TRAIN_RATIO)

        train_files = [audio_files[i] for i in indices[:split_idx]]
        val_files = [audio_files[i] for i in indices[split_idx:]]
        return train_files, val_files

    def write_split_files(
        self, train_files: List[str], val_files: List[str]
    ) -> None:
        """写入验证集和训练集列表文件"""
        val_path = os.path.join(self.base_dir, "validation_list.txt")
        with open(val_path, "w", encoding="utf-8") as f:
            for file in val_files:
                f.write(file + "\n")

        test_path = os.path.join(self.base_dir, "testing_list.txt")
        with open(test_path, "w", encoding="utf-8") as f:
            for file in train_files:
                f.write(file + "\n")

    def update_dataset(self) -> Tuple[int, int]:
        """
        执行完整的数据集更新流程：划分并写入文件。
        返回 (train_count, val_count)
        """
        train_files, val_files = self.split_dataset()
        self.write_split_files(train_files, val_files)
        return len(train_files), len(val_files)

    def export_dataset(self, output_path: str) -> bool:
        """
        将数据集导出为 ZIP 文件。
        返回是否成功。
        """
        try:
            with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                for root, dirs, files in os.walk(self.base_dir):
                    for file in files:
                        if file.endswith(AUDIO_FILE_EXTENSION) or file.endswith(
                            TEXT_FILE_EXTENSION
                        ):
                            full_path = os.path.join(root, file)
                            arcname = os.path.relpath(full_path, self.base_dir)
                            zipf.write(full_path, arcname)
            return True
        except Exception:
            return False

    def set_base_dir(self, new_dir: str) -> None:
        """切换数据集根目录"""
        self.base_dir = new_dir
        os.makedirs(self.base_dir, exist_ok=True)
