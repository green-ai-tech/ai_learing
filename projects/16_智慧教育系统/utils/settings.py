"""系统配置模块

基于 pydantic-settings 的环境配置管理，自动加载 .env 文件。
支持 DeepSeek / Ollama 本地 / Ollama 局域网三种大模型提供商一键切换。

Author: LogicYe
Date: 2026-05-19
"""
from typing import Optional, Literal
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    """系统配置类

    基于 pydantic-settings 的配置管理，自动从 .env 文件加载环境变量。
    通过 llm_provider 字段实现 DeepSeek / Ollama 本地 / Ollama 局域网的一键切换，
    各提供商的 model、api_key、base_url 独立配置互不干扰。
    """
    model_config = SettingsConfigDict(
        env_file=".env",  # 配置文件：不修改py代码的情况下，修改配置，把配置放在python中，访问简单
        env_file_encoding="utf-8",
        case_sensitive=False,    # 大小写不敏感
        extra="ignore"           # 忽略额外的环境变量
        
    )
    # ============ 1. 模型与代理配置 ===============
    # --- 当前生效的提供商 ---
    llm_provider: Literal["deepseek", "ollama_local", "ollama_lan"] = Field(
        default="ollama_local",
        description="当前使用的 LLM 提供商"
    )

    # --- 通用参数 ---
    llm_temperature: float          = Field(default=0.7, ge=0.0, le=2.0, description="模型温度参数")
    llm_max_tokens: Optional[int]   = Field(default=512, description="最大生成 token 数")
    llm_streaming: bool             = Field(default=False, description="是否默认启用流式输出")
    llm_timeout: int                = Field(default=60, description="请求超时秒数")

    # --- DeepSeek ---
    deepseek_model: str             = Field(default="deepseek-chat", description="DeepSeek 模型名")
    deepseek_api_key: Optional[str] = Field(default=None, description="DeepSeek API 密钥")
    deepseek_base_url: str          = Field(default="https://api.deepseek.com/v1", description="DeepSeek 接口地址")

    # --- Ollama 本地 ---
    ollama_local_model: str         = Field(default="qwen3:4b", description="Ollama 本地模型名")
    ollama_local_base_url: str      = Field(default="http://127.0.0.1:11434/v1", description="Ollama 本地地址")

    # --- Ollama 局域网 ---
    ollama_lan_model: str           = Field(default="qwen3:4b", description="Ollama 局域网模型名")
    ollama_lan_base_url: str        = Field(default="http://192.168.1.100:11434/v1", description="Ollama 局域网地址")
    
    # ============ 2. 日志系统的配置 ===============
    log_level: str                  = Field(default="INFO", description="日志级别：DEBUG, INFO, WARNING, ERROR, CRITICAL")
    log_file: str                   = Field(default="logs/stock_agent.log", description="日志文件路径")
    log_rotation: str               = Field(default="100 MB", description="日志文件轮转大小")
    log_retention: str              = Field(default="30 days", description="日志文件保留时间")

# python的编程模式：工厂模式（不使用构造器创建对象，而是使用函数创建对象）
"""全局配置实例

工厂模式创建的单例，项目所有模块通过 import 此实例获取配置。
"""
settings = Settings()

