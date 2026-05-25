from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

from backend.business.apps import APP_STAGE_OPTIONS, list_apps

# 初始化 FastAPI 应用
app = FastAPI(
    title="AI Apps Service",
    description="提供 AI 应用列表数据的 API 服务",
    version="1.0.0"
)


# --- 1. 定义数据模型 (Pydantic) ---
class AppItem(BaseModel):
    id: int
    title: str
    description: str
    stage: str
    tag: str
    icon: str  # 注意：这里将组件对象转为字符串名称，方便前端处理
    iconTone: str
    tagTone: str
    resource: str
    showAction: bool


# --- 3. 定义路由 ---
@app.get("/app-stages", response_model=List[str], summary="获取教学阶段筛选项")
async def get_app_stages():
    return APP_STAGE_OPTIONS


@app.get("/apps", response_model=List[AppItem], summary="获取应用列表")
@app.post("/apps", response_model=List[AppItem], summary="获取应用列表 (POST)")
async def get_apps(
    tag_filter: Optional[str] = Query(None, description="可选：根据标签筛选，例如 '教学前'、'教学中'、'教学后' 或 '其他'"),
    stage_filter: Optional[str] = Query(None, description="可选：根据教学阶段筛选，例如 '教学前'、'教学中'、'教学后' 或 '其他'")
):
    """
    同时支持 GET 和 POST 请求获取应用数据。
    - GET /apps?stage_filter=教学前
    - POST /apps?stage_filter=教学前
    """
    return list_apps(tag_filter=tag_filter, stage_filter=stage_filter)


# --- 4. 启动入口 ---
if __name__ == "__main__":
    # 启动服务，开启自动重载
    uvicorn.run("frontend.eduagents:app", host="0.0.0.0", port=9999, reload=True)
