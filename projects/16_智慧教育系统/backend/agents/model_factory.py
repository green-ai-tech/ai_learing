"""LLM 模型工厂

根据配置中的 llm_provider 参数动态创建对应的 ChatModel 实例：
    - deepseek: ChatOpenAI + DeepSeek API
    - ollama_local: ChatOllama + 本地服务
    - ollama_lan: ChatOllama + 局域网服务

Author: LogicYe
Date: 2026-05-19
"""
from utils.settings import settings


def create_chat_model():
    """创建 ChatModel 实例

    根据 settings.llm_provider 配置动态选择并实例化对应的 ChatModel：
      - deepseek: ChatOpenAI 连接 DeepSeek API
      - ollama_local: ChatOllama 连接本地服务
      - ollama_lan: ChatOllama 连接局域网服务

    Returns:
        BaseChatModel 实例（ChatOpenAI 或 ChatOllama）

    Raises:
        ValueError: 当 llm_provider 值为不支持的提供商时
    """
    provider = settings.llm_provider

    if provider == "deepseek":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=settings.deepseek_model,
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            timeout=settings.llm_timeout,
        )

    if provider == "ollama_local":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=settings.ollama_local_model,
            base_url=settings.ollama_local_base_url,
            temperature=settings.llm_temperature,
        )

    if provider == "ollama_lan":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=settings.ollama_lan_model,
            base_url=settings.ollama_lan_base_url,
            temperature=settings.llm_temperature,
        )

    raise ValueError(f"不支持的 llm_provider: {provider}")
