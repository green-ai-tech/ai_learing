# 语音控制系统 (Voice Controller)

通过语音指令控制系统操作（移动鼠标、截屏、打开计算器等）

## 项目结构

```
07_system_control/
├── config/                     # 配置层
│   ├── __init__.py
│   └── settings.py            # 应用配置（所有常量参数）
│
├── models/                     # 模型层
│   ├── __init__.py
│   └── speech_model.py         # CNN 音频分类模型
│
├── services/                   # 服务层
│   ├── __init__.py
│   ├── audio_recorder.py       # 音频录制与识别服务
│   └── action_executor.py      # 系统动作执行服务
│
├── ui/                          # 表现层
│   ├── __init__.py
│   └── main_window.py          # 主窗口 UI
│
├── old/...                      #旧代码，面向过程
├── run.py                       # 主程序入口

```

## 架构设计

### 分层架构

```
┌─────────────────────────────────────────┐
│         Presentation Layer (UI)         │
│   - MainWindow: 用户界面和交互           │
└─────────────────┬───────────────────────┘
                  │ 依赖注入
┌─────────────────▼───────────────────────┐
│           Service Layer                 │
│   - AudioRecorder: 录音和识别            │
│   - ActionExecutor: 执行系统操作         │
└─────────────────┬───────────────────────┘
                  │ 使用
┌─────────────────▼───────────────────────┐
│           Model Layer                   │
│   - SimpleCNNForAudioClassification     │
└─────────────────┬───────────────────────┘
                  │ 读取
┌─────────────────▼───────────────────────┐
│          Configuration Layer            │
│   - AppConfig: 所有常量参数              │
└─────────────────────────────────────────┘
```

### 设计原则

1. **单一职责原则 (SRP)**: 每个类只有一个明确的职责
2. **依赖注入 (DI)**: 组件之间通过构造函数注入，降低耦合
3. **信号槽机制**: 使用 Qt 信号槽实现组件间通信
4. **配置集中化**: 所有常量集中在 `config/settings.py`

## 快速开始

### 1. 安装依赖

```bash
cd /Volumes/AI/ai_learning
source venv/bin/activate
```

### 2. 运行应用

```bash
cd projects/07_system_control
python run.py
```

### 3. 使用方法

- **按住 Q 键**：开始录音
- **松开 Q 键**：停止录音并识别
- **识别成功**：自动执行对应动作

## 支持的动作

| 语音指令 | 动作描述 |
|---------|---------|
| 动鼠标 | 鼠标画圆 + 锯齿形移动 |
| 截屏 | 保存当前屏幕截图 |
| 打开计算器 | 启动系统计算器应用 |

## 配置说明

在 `config/settings.py` 中修改参数：

```python
class AppConfig:
    # 模型路径
    MODEL_PATH = "/path/to/your/model"
    
    # 识别阈值
    CONFIDENCE_THRESHOLD = 0.35
    
    # 录音长度（秒）
    RECORD_DURATION = 1
    
    # 冷却时间（秒）
    COOLDOWN_SECONDS = 0.5
```

## 训练模型

```bash
# 运行训练脚本
python model_training.py
```

训练完成后，模型会保存到 `MODEL_PATH` 指定的目录。

## 验证模型

```bash
# 运行验证脚本
python model_validation.py
```

## 扩展开发

### 添加新动作

1. 在 `services/action_executor.py` 中添加新方法：

```python
def _do_new_action(self):
    """新动作实现"""
    print("执行新动作")
```

2. 在 `run()` 方法中添加分支：

```python
def run(self):
    if self.command == "新指令":
        self._do_new_action()
```

### 自定义 UI

修改 `ui/main_window.py` 中的 `_setup_ui()` 方法。

## 技术栈

- **Python 3.x**
- **PyTorch**: 深度学习框架
- **Hugging Face Transformers**: 模型格式标准
- **PySide6**: GUI 框架
- **PyAudio**: 音频录制
- **PyAutoGUI**: 系统控制
- **torchaudio**: 音频处理

## 作者

Logic Ye | 2026-04-04

## 许可

本项目为学习用途，未经授权请勿用于商业目的。
