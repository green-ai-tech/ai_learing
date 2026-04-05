# Voice Controller - Architecture Documentation

作者：Logic Ye
日期：2026-04-04

## 1. 项目结构图

```
07_system_control/
│
├── 📁 config/                     [配置层 - 所有常量参数]
│   ├── __init__.py
│   └── settings.py               → AppConfig 类
│
├── 📁 models/                     [模型层 - 机器学习模型]
│   ├── __init__.py
│   └── speech_model.py           → SimpleCNNForAudioClassification
│
├── 📁 services/                   [服务层 - 业务逻辑]
│   ├── __init__.py
│   ├── audio_recorder.py         → AudioRecorder (录音+识别)
│   └── action_executor.py        → ActionExecutor (执行动作)
│
├── 📁 ui/                         [表现层 - 用户界面]
│   ├── __init__.py
│   └── main_window.py            → MainWindow (GUI)
│
├── run.py                         [主程序入口]
├── voice_controller.py            [旧版单文件 - 保留]
├── model_training.py              [训练脚本]
├── model_validation.py            [验证脚本]
└── README.md                      [项目文档]
```

## 2. 类关系图

```
┌─────────────────────────────────────────────────────────┐
│                        AppConfig                         │
│  ─────────────────────────────────────────────────────  │
│  + MODEL_PATH: str                                      │
│  + SAMPLE_RATE: int                                     │
│  + CONFIDENCE_THRESHOLD: float                          │
│  + ENERGY_THRESHOLD: float                              │
│  + COOLDOWN_SECONDS: float                              │
│  + RECORD_DURATION: int                                 │
│  + AUDIO_CHUNK_SIZE: int                                │
│  + N_MELS, N_FFT, HOP_LENGTH: int                       │
│  + WINDOW_WIDTH, WINDOW_HEIGHT: int                     │
│  + FONT_FAMILY: str                                     │
│  + STATUS_COLORS: dict                                  │
│  + MOUSE_CIRCLE_RADIUS: int                             │
│  + MOUSE_ZIGZAG_AMPLITUDE: int                          │
│  + SCREENSHOT_DIR: str                                  │
└──────────────────┬──────────────────────────────────────┘
                   │ 使用
                   ▼
┌─────────────────────────────────────────────────────────┐
│              SimpleCNNForAudioClassification             │
│  ─────────────────────────────────────────────────────  │
│  Inherits: PreTrainedModel (Hugging Face)               │
│  ─────────────────────────────────────────────────────  │
│  + config: PretrainedConfig                             │
│  + num_labels: int                                      │
│  + conv: Sequential                                     │
│  + classifier: Linear                                   │
│  ─────────────────────────────────────────────────────  │
│  + __init__(config)                                     │
│  + forward(input_values, labels, return_dict)           │
└──────────────────┬──────────────────────────────────────┘
                   │ 使用
                   ▼
┌─────────────────────────────────────────────────────────┐
│                    AudioRecorder                         │
│  ─────────────────────────────────────────────────────  │
│  Inherits: QThread                                      │
│  ─────────────────────────────────────────────────────  │
│  Signals:                                               │
│    + result_signal(command: str, confidence: float)     │
│    + status_signal(status: str)                         │
│  ─────────────────────────────────────────────────────  │
│  + model_path: str                                      │
│  + sample_rate: int                                     │
│  + recording: bool                                      │
│  + audio_frames: list                                   │
│  + cooldown: bool                                       │
│  ─────────────────────────────────────────────────────  │
│  + __init__(model_path, sample_rate)                    │
│  + start_recording()                                    │
│  + stop_recording_and_predict()                         │
│  + cleanup()                                            │
│  - _load_model()                                        │
│  - _init_mel_transform()                                │
│  - _audio_callback(in_data, ...)                        │
│  - _predict(audio)                                      │
│  - _clear_cooldown()                                    │
└──────────────────┬──────────────────────────────────────┘
                   │ 使用
                   ▼
┌─────────────────────────────────────────────────────────┐
│                   ActionExecutor                         │
│  ─────────────────────────────────────────────────────  │
│  Inherits: QThread                                      │
│  ─────────────────────────────────────────────────────  │
│  Signals:                                               │
│    + finished()                                         │
│  ─────────────────────────────────────────────────────  │
│  + command: str                                         │
│  ─────────────────────────────────────────────────────  │
│  + __init__(command)                                    │
│  + run()                                                │
│  - _do_mouse_animation()                                │
│  - _draw_circle(...)                                    │
│  - _draw_zigzag(...)                                    │
│  - _do_screenshot()                                     │
│  - _open_calculator()                                   │
└──────────────────┬──────────────────────────────────────┘
                   │ 使用
                   ▼
┌─────────────────────────────────────────────────────────┐
│                     MainWindow                           │
│  ─────────────────────────────────────────────────────  │
│  Inherits: QMainWindow                                  │
│  ─────────────────────────────────────────────────────  │
│  + recorder: AudioRecorder                              │
│  + executor: ActionExecutor                             │
│  + recording_active: bool                               │
│  + result_label: QLabel                                 │
│  + status_label: QLabel                                 │
│  ─────────────────────────────────────────────────────  │
│  + __init__(recorder)                                   │
│  + keyPressEvent(event)                                 │
│  + keyReleaseEvent(event)                               │
│  + closeEvent(event)                                    │
│  - _setup_ui()                                          │
│  - _create_label(...)                                   │
│  - _start_recording()                                   │
│  - _stop_recording_and_predict()                        │
│  - _on_result(command, confidence)  [Slot]              │
│  - _on_status(status)               [Slot]              │
│  - _on_action_finished()            [Slot]              │
└─────────────────────────────────────────────────────────┘
```

## 3. 数据流图

```
用户按键 (Q键)
    │
    ▼
┌──────────────┐
│  MainWindow  │
│  keyPressEvent│
└──────┬───────┘
       │ 调用
       ▼
┌──────────────────────────────────────────────┐
│              AudioRecorder                    │
│  ─────────────────────────────────────────── │
│  1. 打开音频流                                │
│  2. 持续接收音频帧 (_audio_callback)          │
│  3. 用户松开Q键时停止                         │
└──────┬───────────────────────────────────────┘
       │ 合并音频帧
       ▼
┌──────────────────────────────────────────────┐
│              AudioRecorder                    │
│  ─────────────────────────────────────────── │
│  4. 能量检测 (过滤静音)                       │
│  5. 转换为 Mel 频谱图                         │
│  6. 标准化                                    │
│  7. 模型推理 (SimpleCNNForAudioClassification)│
│  8. 获取预测结果                              │
└──────┬───────────────────────────────────────┘
       │ 发送信号 result_signal(command, confidence)
       ▼
┌──────────────────────────────────────────────┐
│              MainWindow                       │
│  ─────────────────────────────────────────── │
│  _on_result(command, confidence)              │
│  ─────────────────────────────────────────── │
│  1. 更新 UI 显示结果                          │
│  2. 创建 ActionExecutor                      │
│  3. 启动执行器                                │
└──────┬───────────────────────────────────────┘
       │ 调用
       ▼
┌──────────────────────────────────────────────┐
│             ActionExecutor                    │
│  ─────────────────────────────────────────── │
│  run()                                        │
│  ─────────────────────────────────────────── │
│  根据 command 执行对应动作:                   │
│    - 动鼠标: _do_mouse_animation()            │
│    - 截屏: _do_screenshot()                   │
│    - 打开计算器: _open_calculator()           │
└──────┬───────────────────────────────────────┘
       │ 发送信号 finished()
       ▼
┌──────────────────────────────────────────────┐
│              MainWindow                       │
│  ─────────────────────────────────────────── │
│  _on_action_finished()                        │
│  (恢复监听状态)                               │
└──────────────────────────────────────────────┘
```

## 4. 模块依赖关系

```
run.py
  │
  ├─→ config/settings.py
  │     └── AppConfig (所有模块共享配置)
  │
  ├─→ services/audio_recorder.py
  │     ├─→ config/settings.py
  │     ├─→ models/speech_model.py
  │     └─→ PySide6.QtCore (QThread, Signal, QTimer)
  │
  ├─→ ui/main_window.py
  │     ├─→ config/settings.py
  │     ├─→ services/audio_recorder.py
  │     ├─→ services/action_executor.py
  │     └─→ PySide6.QtWidgets, QtCore, QtGui
  │
  └─→ services/action_executor.py
        ├─→ config/settings.py
        ├─→ pyautogui
        ├─→ PIL.ImageGrab
        └─→ subprocess, sys
```

## 5. 执行流程

```
启动应用
  │
  ▼
1. main() 函数 (run.py)
  │
  ├─→ create_app()           → 创建 QApplication
  ├─→ create_recorder()      → 创建 AudioRecorder (加载模型)
  ├─→ recorder.start()       → 启动后台线程
  └─→ create_window()        → 创建 MainWindow (注入 recorder)
       │
       ▼
2. 用户交互循环
  │
  ├─→ 用户按Q键 → keyPressEvent() → _start_recording()
  │                    │
  │                    ├─→ recorder.start_recording()
  │                    └─→ 开始录音
  │
  ├─→ 用户松开Q键 → keyReleaseEvent() → _stop_recording_and_predict()
  │                    │
  │                    ├─→ recorder.stop_recording_and_predict()
  │                    │     │
  │                    │     ├─→ 能量检测
  │                    │     ├─→ 模型推理
  │                    │     └─→ 发射 result_signal
  │                    │
  │                    └─→ _on_result(command, confidence)
  │                          │
  │                          ├─→ 更新 UI
  │                          └─→ ActionExecutor(command).start()
  │                                │
  │                                ├─→ 执行动作
  │                                └─→ 发射 finished
  │                                      │
  │                                      └─→ _on_action_finished()
  │
  ▼
3. 用户关闭窗口 → closeEvent() → recorder.cleanup()
```

## 6. 设计模式应用

| 设计模式 | 应用场景 | 优势 |
|---------|---------|------|
| **分层架构** | config → models → services → ui | 职责清晰，易于维护 |
| **依赖注入** | MainWindow 接收 AudioRecorder | 降低耦合，便于测试 |
| **观察者模式** | Qt 信号槽机制 | 松耦合的组件通信 |
| **单例模式** | AppConfig (类属性) | 集中管理配置 |
| **工厂方法** | create_app(), create_recorder() | 统一对象创建逻辑 |
| **策略模式** | ActionExecutor 的不同动作 | 易于扩展新动作 |

## 7. 扩展指南

### 添加新语音指令

**步骤 1**: 在 `config/settings.py` 中确认配置（如需要）

**步骤 2**: 在 `services/action_executor.py` 的 `run()` 方法中添加分支：

```python
def run(self):
    if self.command == "动鼠标":
        self._do_mouse_animation()
    elif self.command == "截屏":
        self._do_screenshot()
    elif self.command == "打开计算器":
        self._open_calculator()
    elif self.command == "新指令":  # ← 添加这里
        self._do_new_action()      # ← 调用新方法
    self.finished.emit()

def _do_new_action(self):
    """新动作实现"""
    print("执行新动作")
```

**步骤 3**: 训练模型识别新指令（参考 `model_training.py`）

### 自定义 UI 样式

修改 `ui/main_window.py` 的 `_setup_ui()` 方法或 `AppConfig` 中的颜色配置。

### 添加新模型

1. 在 `models/` 下创建新文件
2. 继承 `PreTrainedModel`
3. 在 `services/audio_recorder.py` 中替换模型加载逻辑
