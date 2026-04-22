#MCP:Model Context Protocol
#环境：pip install fastmcp
from fastmcp import FastMCP

mcp = FastMCP("Math")   

@mcp.tool
def add(a:int,b:int)->int:
    """将a与b相加,返回和"""
    return a+b


@mcp.tool
def mulity(a:int,b:int)->int:
    """将a与b相乘,返回和"""
    return a*b


#发布服务

if __name__ == "__mian__":
    mcp.run()







