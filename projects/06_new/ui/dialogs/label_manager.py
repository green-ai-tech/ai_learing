#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
标签管理对话框 - 用于新增或删除标签
"""

import os
from typing import List, Optional

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout as QHBoxLayoutDlg,
    QInputDialog,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton as QPushButtonDlg,
    QVBoxLayout as QVBoxLayoutDlg,
)

from config import DEFAULT_LABELS


class LabelManageDialog(QDialog):
    """自定义标签管理对话框"""

    def __init__(
        self, current_labels: List[str], base_dir: str, parent: Optional[QDialog] = None
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("自定义标签管理")
        self.setMinimumWidth(300)
        self.base_dir = base_dir
        self.all_labels = current_labels.copy()
        self.init_ui()

    def init_ui(self) -> None:
        layout = QVBoxLayoutDlg(self)

        self.list_widget = QListWidget()
        for label in self.all_labels:
            self.list_widget.addItem(label)
        layout.addWidget(self.list_widget)

        btn_layout = QHBoxLayoutDlg()
        self.add_btn = QPushButtonDlg("新增标签")
        self.del_btn = QPushButtonDlg("删除选中")
        btn_layout.addWidget(self.add_btn)
        btn_layout.addWidget(self.del_btn)
        layout.addLayout(btn_layout)

        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self.add_btn.clicked.connect(self.add_label)
        self.del_btn.clicked.connect(self.delete_label)

    def add_label(self) -> None:
        """新增标签"""
        new_label, ok = QInputDialog.getText(self, "新增标签", "请输入新标签名称:")
        if ok and new_label.strip():
            new_label = new_label.strip()
            if new_label not in self.all_labels:
                self.all_labels.append(new_label)
                self.list_widget.addItem(new_label)
                label_dir = os.path.join(self.base_dir, new_label)
                os.makedirs(label_dir, exist_ok=True)
                QMessageBox.information(
                    self, "成功", f"标签 '{new_label}' 已添加，文件夹已创建"
                )
            else:
                QMessageBox.warning(self, "警告", "标签已存在")

    def delete_label(self) -> None:
        """删除选中标签"""
        current = self.list_widget.currentItem()
        if not current:
            QMessageBox.warning(self, "警告", "请先选中要删除的标签")
            return
        label = current.text()
        reply = QMessageBox.question(
            self,
            "确认删除",
            f"确定要删除标签 '{label}' 吗？\n对应的文件夹也会被删除（如果有音频文件将丢失）",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.all_labels.remove(label)
            self.list_widget.takeItem(self.list_widget.row(current))
            label_dir = os.path.join(self.base_dir, label)
            if os.path.exists(label_dir):
                import shutil

                shutil.rmtree(label_dir)
            QMessageBox.information(self, "成功", f"标签 '{label}' 已删除")

    def get_labels(self) -> List[str]:
        """获取更新后的标签列表"""
        return self.all_labels
