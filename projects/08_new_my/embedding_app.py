"""
本地文件一键向量化 + 向量数据库存储 + 词嵌入可视化桌面应用
技术栈: PySide6, Chroma, ollama, PyPDF2, matplotlib, scikit-learn
启动: python embedding_app.py
安装依赖: pip install PySide6 chromadb requests pypdf numpy matplotlib scikit-learn
"""

import sys
import os
import re
import json
import uuid
import traceback
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict

import numpy as np
import requests
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QTextEdit, QTableWidget,
    QTableWidgetItem, QProgressBar, QFileDialog, QMessageBox,
    QHeaderView, QSplitter, QGroupBox, QSizePolicy
)
from PySide6.QtCore import QThread, Signal, Qt
from PySide6.QtGui import QFont, QColor, QTextCursor

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['PingFang SC', 'Arial Unicode MS', 'SimHei', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.cm as cm

from sklearn.decomposition import PCA

import chromadb
from chromadb.config import Settings

# ============================ 文本清洗 ============================

def clean_text(raw: str) -> str:
    """对原始文本执行清洗：去多余空格/换行/特殊字符/分段"""
    if not raw or not raw.strip():
        return ""

    # 统一换行为 \n
    text = raw.replace("\r\n", "\n").replace("\r", "\n")

    # 去除制表符，替换为空格
    text = text.replace("\t", " ")

    # 去除不可见字符（保留空格和换行）
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

    # 去除连续多个空格
    text = re.sub(r'[ ]{2,}', ' ', text)

    # 去除连续多个换行（最多保留 1 个空行）
    text = re.sub(r'\n{3,}', '\n\n', text)

    # 去除行首行尾空格
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)

    # 统一中英文标点（简单处理）
    text = text.replace("，", ",").replace("。", ".").replace("！", "!").replace("？", "?")
    text = text.replace("；", ";").replace("：", ":").replace("（", "(").replace("）", ")")

    # 按段落切分，过滤过短段落（< 10 字符）
    paragraphs = [p.strip() for p in text.split("\n") if len(p.strip()) >= 10]
    return "\n".join(paragraphs)


def split_into_chunks(text: str, max_len: int = 500) -> List[str]:
    """将清洗后的文本按句子/段落切分为适合 embedding 的片段"""
    if not text:
        return []

    # 按句子切（中英文句号、问号、感叹号）
    sentences = re.split(r'(?<=[.!?。！？])\s*', text)
    chunks = []
    current = ""
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if len(current) + len(s) + 1 <= max_len:
            current = current + " " + s if current else s
        else:
            if current and len(current) >= 10:
                chunks.append(current)
            # 如果单句超长，强制截断
            if len(s) > max_len:
                for i in range(0, len(s), max_len):
                    chunks.append(s[i:i + max_len])
                current = ""
            else:
                current = s
    if current and len(current) >= 10:
        chunks.append(current)
    return chunks


# ============================ Ollama Embedding ============================

def get_ollama_embedding(text: str, model: str = "qwen3-embedding:4b",
                         base_url: str = "http://localhost:11434") -> List[float]:
    """调用 ollama API 获取文本 embedding"""
    url = f"{base_url}/api/embeddings"
    payload = {"model": model, "prompt": text}
    try:
        resp = requests.post(url, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return data.get("embedding", [])
    except requests.exceptions.ConnectionError:
        raise RuntimeError(f"无法连接到 Ollama 服务 ({base_url})，请确认 ollama 已启动。")
    except Exception as e:
        raise RuntimeError(f"获取 embedding 失败: {e}")


# ============================ 后台任务线程 ============================

class VectorizationWorker(QThread):
    """后台向量化线程，不阻塞 UI"""
    log = Signal(str)
    progress = Signal(int, int)  # current, total
    file_scanned = Signal(list)  # list of file info dicts
    embeddings_ready = Signal(list, list)  # vectors, labels
    error = Signal(str)
    finished_ok = Signal(int)  # total chunks stored

    def __init__(self, source_dir: str, db_path: str, embed_model: str, embed_url: str):
        super().__init__()
        self.source_dir = source_dir
        self.db_path = db_path
        self.embed_model = embed_model
        self.embed_url = embed_url
        self._running = True

    def stop(self):
        self._running = False

    def run(self):
        try:
            # 1. 扫描文件
            self.log.emit("🔍 开始扫描文件...")
            files = self._scan_files()
            if not files:
                self.log.emit("⚠️ 未找到任何 PDF/TXT/MD 文件")
                return
            self.file_scanned.emit(files)
            self.log.emit(f"📄 共找到 {len(files)} 个文件")

            # 2. 初始化 Chroma 向量库
            self.log.emit(f"🗄️ 初始化向量数据库: {self.db_path}")
            client = chromadb.PersistentClient(path=self.db_path)
            collection_name = "documents"
            try:
                collection = client.get_collection(collection_name)
                self.log.emit(f"📂 使用已存在的集合: {collection_name}")
            except Exception:
                collection = client.create_collection(collection_name)
                self.log.emit(f"✅ 创建新集合: {collection_name}")

            # 3. 处理每个文件
            total_chunks = 0
            all_vectors = []
            all_labels = []

            for idx, finfo in enumerate(files):
                if not self._running:
                    self.log.emit("⏹️ 用户取消操作")
                    return

                fname = finfo["name"]
                self.log.emit(f"\n📖 处理 [{idx+1}/{len(files)}]: {fname}")
                self.progress.emit(idx, len(files))

                # 读取文本
                raw_text = self._read_file(finfo["path"])
                if not raw_text:
                    self.log.emit(f"  ⚠️ 文件内容为空，跳过")
                    continue

                # 清洗
                cleaned = clean_text(raw_text)
                if not cleaned:
                    self.log.emit(f"  ⚠️ 清洗后无有效内容，跳过")
                    continue

                # 分块
                chunks = split_into_chunks(cleaned)
                if not chunks:
                    self.log.emit(f"  ⚠️ 无法分块，跳过")
                    continue

                self.log.emit(f"  📝 清洗完成，分为 {len(chunks)} 个片段")

                # 向量化 + 入库
                ids = []
                metas = []
                docs = []
                vectors_for_viz = []
                for cidx, chunk in enumerate(chunks):
                    if not self._running:
                        self.log.emit("⏹️ 用户取消操作")
                        return

                    try:
                        vec = get_ollama_embedding(chunk, self.embed_model, self.embed_url)
                    except Exception as e:
                        self.log.emit(f"  ❌ 第 {cidx+1} 片段 embedding 失败: {e}")
                        continue

                    doc_id = f"{finfo['name']}_{cidx}"
                    meta = {
                        "file_name": fname,
                        "file_path": str(finfo["path"]),
                        "chunk_index": cidx,
                        "file_type": finfo["type"],
                        "file_size": finfo["size"],
                        "processed_at": datetime.now().isoformat(),
                    }

                    ids.append(doc_id)
                    metas.append(meta)
                    docs.append(chunk)
                    vectors_for_viz.append(vec)
                    total_chunks += 1

                if ids:
                    # 分批插入（Chroma 单次上限）
                    batch_size = 50
                    for start in range(0, len(ids), batch_size):
                        end = min(start + batch_size, len(ids))
                        collection.add(
                            ids=ids[start:end],
                            documents=docs[start:end],
                            metadatas=metas[start:end],
                            embeddings=vectors_for_viz[start:end],
                        )
                    self.log.emit(f"  ✅ 成功入库 {len(ids)} 个向量")

                    # 收集可视化数据
                    all_vectors.extend(vectors_for_viz)
                    all_labels.extend([f"{fname} (x{len(ids)})"] * len(ids))

            self.log.emit(f"\n🎉 全部完成！共存储 {total_chunks} 个向量片段")
            self.embeddings_ready.emit(all_vectors, all_labels)
            self.finished_ok.emit(total_chunks)

        except Exception as e:
            self.error.emit(f"❌ 处理失败: {e}\n{traceback.format_exc()}")

    def _scan_files(self) -> List[Dict]:
        """扫描目录，返回文件信息列表"""
        valid_ext = {".pdf", ".txt", ".md", ".markdown"}
        files = []
        for p in Path(self.source_dir).rglob("*"):
            if p.suffix.lower() in valid_ext and p.is_file():
                files.append({
                    "name": p.name,
                    "path": p,
                    "type": p.suffix.lower(),
                    "size": self._fmt_size(p.stat().st_size),
                    "size_bytes": p.stat().st_size,
                })
        return sorted(files, key=lambda x: x["size_bytes"], reverse=True)

    def _fmt_size(self, n: int) -> str:
        if n < 1024:
            return f"{n} B"
        elif n < 1024 * 1024:
            return f"{n / 1024:.1f} KB"
        else:
            return f"{n / (1024 * 1024):.1f} MB"

    def _read_file(self, path: Path) -> str:
        """读取文件内容"""
        suffix = path.suffix.lower()
        try:
            if suffix == ".pdf":
                # 1. 尝试 pypdf
                try:
                    from pypdf import PdfReader
                    with open(path, "rb") as f:
                        reader = PdfReader(f)
                        if len(reader.pages) == 0:
                            return ""
                        text = ""
                        for page in reader.pages:
                            page_text = page.extract_text()
                            if page_text:
                                text += page_text + "\n"
                    if not text.strip():
                        self.log.emit(f"  ⚠️ PDF 可能为扫描图片版，无法提取文本")
                    return text
                except Exception as e:
                    self.log.emit(f"  ⚠️ pypdf 解析失败 ({type(e).__name__})")

                # 2. 尝试 PyPDF2
                try:
                    import PyPDF2
                    with open(path, "rb") as f:
                        reader = PyPDF2.PdfReader(f)
                        if len(reader.pages) == 0:
                            return ""
                        text = ""
                        for page in reader.pages:
                            page_text = page.extract_text()
                            if page_text:
                                text += page_text + "\n"
                    return text
                except Exception:
                    pass

                # 3. 尝试 pdfplumber (对复杂排版兼容性更好)
                try:
                    import pdfplumber
                    with pdfplumber.open(path) as pdf:
                        if not pdf.pages:
                            return ""
                        text = ""
                        for page in pdf.pages:
                            t = page.extract_text()
                            if t:
                                text += t + "\n"
                    return text
                except ImportError:
                    self.log.emit("  💡 提示: 安装 `pdfplumber` 可能解决复杂 PDF 读取问题")
                    return ""
                except Exception:
                    return ""
            else:
                # TXT / MD
                for enc in ["utf-8", "gbk", "gb2312", "latin-1"]:
                    try:
                        with open(path, "r", encoding=enc) as f:
                            return f.read()
                    except UnicodeDecodeError:
                        continue
                return ""
        except Exception as e:
            self.log.emit(f"  ⚠️ 读取失败: {e}")
            return ""


# ============================ 可视化面板 ============================

class EmbeddingVisualizer(QWidget):
    """PCA 降维散点图"""

    def __init__(self):
        super().__init__()
        self.figure = Figure(figsize=(10, 4), dpi=100, facecolor="#1e1e1e")
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)

        self._tooltip = None
        self.canvas.mpl_connect("motion_notify_event", self._on_hover)

    def plot(self, vectors: List[List[float]], labels: List[str]):
        """绘制 PCA 降维散点图"""
        self.figure.clear()

        # 暗色主题样式
        self.figure.set_facecolor("#1e1e1e")
        self.canvas.setStyleSheet("background-color: #1e1e1e;")

        if not vectors or len(vectors) < 2:
            ax = self.figure.add_subplot(111)
            ax.set_facecolor("#1e1e1e")
            ax.text(0.5, 0.5, "向量数量不足，无法可视化", color="#aaa", ha="center", va="center", fontsize=14)
            ax.axis("off")
            self.figure.tight_layout()
            self.canvas.draw()
            return

        # PCA 降维到 2D
        X = np.array(vectors)
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X)

        ax = self.figure.add_subplot(111)
        ax.set_facecolor("#1e1e1e")

        # 按文件名分组着色
        file_groups: Dict[str, list] = {}
        for i, label in enumerate(labels):
            fname = label.split(" (")[0]
            file_groups.setdefault(fname, []).append((X_2d[i, 0], X_2d[i, 1]))

        colors = cm.tab20(np.linspace(0, 1, max(len(file_groups), 1)))
        for cidx, (fname, points) in enumerate(file_groups.items()):
            pts = np.array(points)
            ax.scatter(pts[:, 0], pts[:, 1], c=[colors[cidx % len(colors)]],
                       label=fname[:30], alpha=0.8, s=40, edgecolors="#fff", linewidths=0.5)

        variance = sum(pca.explained_variance_ratio_)
        ax.set_title(f"Embedding PCA 可视化 (n={len(vectors)}, 方差解释: {variance:.1%})",
                     color="#eee", fontsize=12)
        ax.set_xlabel("PC1", color="#ccc")
        ax.set_ylabel("PC2", color="#ccc")
        ax.tick_params(colors="#aaa")
        legend = ax.legend(loc="upper right", fontsize=8, framealpha=0.3)
        for text in legend.get_texts():
            text.set_color("#ddd")
        ax.grid(True, alpha=0.15, color="#555")

        for spine in ax.spines.values():
            spine.set_color("#555")

        self.figure.tight_layout()
        self.canvas.draw()

    def _on_hover(self, event):
        if event.inaxes is None:
            return
        x, y = event.xdata, event.ydata
        self.canvas.setToolTip(f"({x:.2f}, {y:.2f})")


# ============================ 主窗口 ============================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.worker = None
        self.all_vectors = []
        self.all_labels = []

        self.setWindowTitle("📚 本地文件向量化 + 向量数据库 + 嵌入可视化")
        self.resize(1200, 850)
        self._setup_ui()

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # ===== 顶部：路径设置 =====
        path_group = QGroupBox("📂 路径设置")
        path_layout = QVBoxLayout(path_group)

        # 源文件夹
        src_row = QHBoxLayout()
        src_row.addWidget(QLabel("源文件夹:"))
        self.src_input = QLineEdit("/Users/logicye/Code/ai_learning/assets/pdf")
        src_row.addWidget(self.src_input)
        self.src_btn = QPushButton("浏览...")
        self.src_btn.clicked.connect(lambda: self._browse_folder(self.src_input))
        src_row.addWidget(self.src_btn)
        path_layout.addLayout(src_row)

        # 向量库路径
        db_row = QHBoxLayout()
        db_row.addWidget(QLabel("向量数据库路径:"))
        self.db_input = QLineEdit("/Users/logicye/Code/ai_learning/assets/vector_database")
        db_row.addWidget(self.db_input)
        self.db_btn = QPushButton("浏览...")
        self.db_btn.clicked.connect(lambda: self._browse_folder(self.db_input))
        db_row.addWidget(self.db_btn)
        path_layout.addLayout(db_row)

        # 嵌入模型设置
        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("嵌入模型:"))
        self.model_input = QLineEdit("qwen3-embedding:4b")
        self.model_input.setFixedWidth(250)
        model_row.addWidget(self.model_input)
        model_row.addSpacing(20)
        model_row.addWidget(QLabel("Ollama 地址:"))
        self.url_input = QLineEdit("http://localhost:11434")
        self.url_input.setFixedWidth(220)
        model_row.addWidget(self.url_input)
        model_row.addStretch()
        path_layout.addLayout(model_row)

        main_layout.addWidget(path_group)

        # ===== 按钮区 =====
        btn_layout = QHBoxLayout()

        self.btn_scan = QPushButton("🔍 扫描文件")
        self.btn_scan.clicked.connect(self._scan_files)
        btn_layout.addWidget(self.btn_scan)

        self.btn_run = QPushButton("🚀 一键清洗 + 向量化 + 入库")
        self.btn_run.clicked.connect(self._start_vectorization)
        btn_layout.addWidget(self.btn_run)

        self.btn_stop = QPushButton("⏹️ 停止")
        self.btn_stop.clicked.connect(self._stop_worker)
        self.btn_stop.setEnabled(False)
        btn_layout.addWidget(self.btn_stop)

        self.btn_clear_log = QPushButton("🗑️ 清空日志")
        self.btn_clear_log.clicked.connect(lambda: self.log_box.clear())
        btn_layout.addWidget(self.btn_clear_log)

        self.btn_open_db = QPushButton("📁 打开向量库目录")
        self.btn_open_db.clicked.connect(self._open_db_dir)
        btn_layout.addWidget(self.btn_open_db)

        btn_layout.addStretch()
        main_layout.addLayout(btn_layout)

        # ===== 进度条 =====
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(20)
        main_layout.addWidget(self.progress_bar)

        # ===== 中间：文件列表 + 日志 =====
        splitter = QSplitter(Qt.Horizontal)

        # 左侧：文件列表
        file_group = QGroupBox("📄 文件列表")
        file_layout = QVBoxLayout(file_group)
        self.file_table = QTableWidget()
        self.file_table.setColumnCount(4)
        self.file_table.setHorizontalHeaderLabels(["文件名", "大小", "类型", "状态"])
        self.file_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.file_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Fixed)
        self.file_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Fixed)
        self.file_table.setColumnWidth(1, 80)
        self.file_table.setColumnWidth(2, 60)
        self.file_table.setColumnWidth(3, 60)
        file_layout.addWidget(self.file_table)
        splitter.addWidget(file_group)

        # 右侧：日志
        log_group = QGroupBox("📝 运行日志")
        log_layout = QVBoxLayout(log_group)
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setFont(QFont("Menlo", 10))
        log_layout.addWidget(self.log_box)
        splitter.addWidget(log_group)

        splitter.setSizes([350, 500])
        main_layout.addWidget(splitter)

        # ===== 底部：可视化 =====
        viz_group = QGroupBox("📊 Embedding PCA 可视化")
        viz_layout = QVBoxLayout(viz_group)
        self.visualizer = EmbeddingVisualizer()
        viz_layout.addWidget(self.visualizer)
        main_layout.addWidget(viz_group)

    # ---------- 工具方法 ----------

    def _browse_folder(self, line_edit: QLineEdit):
        path = QFileDialog.getExistingDirectory(self, "选择文件夹")
        if path:
            line_edit.setText(path)

    def _log(self, msg: str):
        self.log_box.append(msg)
        cursor = self.log_box.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.log_box.setTextCursor(cursor)

    def _scan_files(self):
        src = self.src_input.text().strip()
        if not src or not os.path.isdir(src):
            QMessageBox.warning(self, "警告", "请输入有效的源文件夹路径")
            return

        valid_ext = {".pdf", ".txt", ".md", ".markdown"}
        files = []
        for p in Path(src).rglob("*"):
            if p.suffix.lower() in valid_ext and p.is_file():
                size = p.stat().st_size
                if size < 1024:
                    size_str = f"{size} B"
                elif size < 1024 * 1024:
                    size_str = f"{size / 1024:.1f} KB"
                else:
                    size_str = f"{size / (1024 * 1024):.1f} MB"
                files.append({"name": p.name, "size": size_str, "type": p.suffix.lower()})

        self.file_table.setRowCount(len(files))
        for i, f in enumerate(files):
            self.file_table.setItem(i, 0, QTableWidgetItem(f["name"]))
            self.file_table.setItem(i, 1, QTableWidgetItem(f["size"]))
            self.file_table.setItem(i, 2, QTableWidgetItem(f["type"]))
            self.file_table.setItem(i, 3, QTableWidgetItem("待处理"))
            self.file_table.item(i, 3).setTextAlignment(Qt.AlignCenter)

        self._log(f"📄 扫描完成，共找到 {len(files)} 个文件")

    def _start_vectorization(self):
        src = self.src_input.text().strip()
        db = self.db_input.text().strip()
        model = self.model_input.text().strip()
        url = self.url_input.text().strip()

        if not src or not os.path.isdir(src):
            QMessageBox.warning(self, "警告", "请输入有效的源文件夹路径")
            return
        if not db:
            QMessageBox.warning(self, "警告", "请输入向量数据库路径")
            return

        # 禁用按钮
        self.btn_run.setEnabled(False)
        self.btn_scan.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.progress_bar.setValue(0)

        # 创建 worker
        self.worker = VectorizationWorker(src, db, model, url)
        self.worker.log.connect(self._log)
        self.worker.progress.connect(lambda cur, tot: self.progress_bar.setValue(int((cur / tot) * 100) if tot else 0))
        self.worker.file_scanned.connect(self._on_files_scanned)
        self.worker.embeddings_ready.connect(self._on_embeddings_ready)
        self.worker.error.connect(lambda msg: QMessageBox.critical(self, "错误", msg))
        self.worker.error.connect(lambda _: self._reset_buttons())
        self.worker.finished_ok.connect(lambda n: self._reset_buttons())

        self._log("🚀 开始向量化任务...")
        self.worker.start()

    def _stop_worker(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self._log("⏹️ 正在停止...")

    def _reset_buttons(self):
        self.btn_run.setEnabled(True)
        self.btn_scan.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.progress_bar.setValue(100)

    def _on_files_scanned(self, files):
        self.file_table.setRowCount(len(files))
        for i, f in enumerate(files):
            self.file_table.setItem(i, 0, QTableWidgetItem(f["name"]))
            self.file_table.setItem(i, 1, QTableWidgetItem(f["size"]))
            self.file_table.setItem(i, 2, QTableWidgetItem(f["type"]))
            self.file_table.setItem(i, 3, QTableWidgetItem("待处理"))
            self.file_table.item(i, 3).setTextAlignment(Qt.AlignCenter)

    def _on_embeddings_ready(self, vectors, labels):
        self.all_vectors = vectors
        self.all_labels = labels
        if vectors:
            self.visualizer.plot(vectors, labels)
            self._log(f"📊 可视化已更新，共 {len(vectors)} 个向量点")

    def _open_db_dir(self):
        db_path = self.db_input.text().strip()
        if db_path and os.path.isdir(db_path):
            import subprocess
            if sys.platform == "darwin":
                subprocess.run(["open", db_path])
            elif sys.platform == "win32":
                os.startfile(db_path)
            else:
                subprocess.run(["xdg-open", db_path])
        else:
            QMessageBox.information(self, "提示", "向量数据库目录不存在")


# ============================ 入口 ============================

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    font = QFont("PingFang SC", 12)
    app.setFont(font)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
