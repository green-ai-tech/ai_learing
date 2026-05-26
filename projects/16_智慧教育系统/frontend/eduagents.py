"""EduAgent FastAPI 服务入口。"""

from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError

from backend.api.apps import router as apps_router
from backend.api.outline import router as outline_router
from backend.database import init_db
from utils.logger import get_logger, setup_logging


setup_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db(raise_on_error=False)
    yield


app = FastAPI(
    title="EduAgent Service",
    description="提供教学智能体卡片、教学大纲生成、导出与课件生成 API 服务",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(SQLAlchemyError)
async def sqlalchemy_exception_handler(_: Request, exc: SQLAlchemyError):
    logger.exception(f"数据库访问失败: {exc}")
    return JSONResponse(status_code=503, content={"detail": "数据库连接失败，请检查 MySQL 服务与 .env 配置"})


app.include_router(apps_router)
app.include_router(outline_router)
app.include_router(apps_router, prefix="/api")
app.include_router(outline_router, prefix="/api")


if __name__ == "__main__":
    uvicorn.run("frontend.eduagents:app", host="0.0.0.0", port=9999, reload=True)
