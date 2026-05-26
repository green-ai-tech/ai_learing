"""数据库连接与 ORM 基础设施。"""

from __future__ import annotations

from typing import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from utils.logger import get_logger
from utils.settings import settings


logger = get_logger(__name__)


class Base(DeclarativeBase):
    """项目统一 ORM Base。"""


engine = create_engine(
    settings.resolved_database_url,
    pool_pre_ping=True,
    pool_recycle=3600,
    future=True,
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)


def create_database_if_missing() -> None:
    """在 MySQL 中创建目标数据库，避免首次启动手动建库。"""
    url = make_url(settings.resolved_database_url)
    database_name = url.database
    if not database_name:
        return

    server_url = url.set(database=None)
    server_engine = create_engine(server_url, pool_pre_ping=True, future=True)
    with server_engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        conn.execute(
            text(
                f"CREATE DATABASE IF NOT EXISTS `{database_name}` "
                "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
            )
        )
    server_engine.dispose()


def init_db(raise_on_error: bool = False) -> bool:
    """初始化数据库连接、自动建库并创建表。"""
    try:
        create_database_if_missing()

        from backend.models import import_all_models

        import_all_models()
        Base.metadata.create_all(bind=engine)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("MySQL 数据库连接检查通过，表结构已同步")
        return True
    except SQLAlchemyError as exc:
        logger.exception(f"MySQL 数据库初始化失败: {exc}")
        if raise_on_error:
            raise
        return False


def get_db() -> Generator[Session, None, None]:
    """FastAPI 数据库 Session 依赖。"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
