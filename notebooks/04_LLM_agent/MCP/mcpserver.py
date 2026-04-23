# MCP = Model Context Protocol
# 环境: pip install fastmcp

from fastmcp import FastMCP

mcp = FastMCP("Math") # 类似HTTP响应对象，但不准确

@mcp.tool
def add(a: int, b: int) -> int:
    """
    执行加法运算，将`a`和`b`相加
    """
    return a + b

@mcp.tool
def multy(a: int, b: int) -> int:
    """
    执行乘法运算，将`a`和`b`相乘
    """
    return a * b

# 发布成服务
if __name__=="__main__":
    mcp.run(transport="stdio")   # 默认stdio  # 本地进程