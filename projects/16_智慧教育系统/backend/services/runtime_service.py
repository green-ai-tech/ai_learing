"""轻量 Runtime、store、checkpointer 与日志持久化服务。"""

from __future__ import annotations

from typing import Any, Optional

from sqlalchemy.orm import Session

from backend.models.runtime import (
    CheckpointModel,
    ConversationModel,
    RuntimeLogModel,
    StoreRecordModel,
)
from utils.logger import get_logger


logger = get_logger(__name__)


def ensure_conversation(
    db: Session,
    conversation_id: str,
    title: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> ConversationModel:
    conversation = (
        db.query(ConversationModel)
        .filter(ConversationModel.conversation_id == conversation_id)
        .one_or_none()
    )
    if conversation:
        return conversation

    conversation = ConversationModel(
        conversation_id=conversation_id,
        title=title,
        metadata_json=metadata or {},
    )
    db.add(conversation)
    db.flush()
    return conversation


def log_runtime_event(
    db: Session,
    *,
    event: str,
    message: str,
    conversation_id: Optional[str] = None,
    outline_id: Optional[int] = None,
    runtime_name: str = "outline",
    level: str = "INFO",
    payload: Optional[dict[str, Any]] = None,
) -> RuntimeLogModel:
    record = RuntimeLogModel(
        conversation_id=conversation_id,
        outline_id=outline_id,
        runtime_name=runtime_name,
        level=level,
        event=event,
        message=message,
        payload=payload or {},
    )
    db.add(record)
    db.flush()

    log_message = f"{event}: {message}"
    if level.upper() == "ERROR":
        logger.error(log_message)
    elif level.upper() == "WARNING":
        logger.warning(log_message)
    else:
        logger.info(log_message)
    return record


def save_store_record(
    db: Session,
    *,
    conversation_id: str,
    namespace: str,
    record_key: str,
    value: dict[str, Any],
) -> StoreRecordModel:
    record = (
        db.query(StoreRecordModel)
        .filter(
            StoreRecordModel.conversation_id == conversation_id,
            StoreRecordModel.namespace == namespace,
            StoreRecordModel.record_key == record_key,
        )
        .one_or_none()
    )
    if record:
        record.value_json = value
    else:
        record = StoreRecordModel(
            conversation_id=conversation_id,
            namespace=namespace,
            record_key=record_key,
            value_json=value,
        )
        db.add(record)
    db.flush()
    return record


def save_checkpoint(
    db: Session,
    *,
    conversation_id: str,
    checkpoint_name: str,
    state: dict[str, Any],
) -> CheckpointModel:
    record = (
        db.query(CheckpointModel)
        .filter(
            CheckpointModel.conversation_id == conversation_id,
            CheckpointModel.checkpoint_name == checkpoint_name,
        )
        .one_or_none()
    )
    if record:
        record.state_json = state
    else:
        record = CheckpointModel(
            conversation_id=conversation_id,
            checkpoint_name=checkpoint_name,
            state_json=state,
        )
        db.add(record)
    db.flush()
    return record


def list_runtime_logs(
    db: Session,
    *,
    conversation_id: Optional[str] = None,
    outline_id: Optional[int] = None,
    limit: int = 200,
) -> list[RuntimeLogModel]:
    query = db.query(RuntimeLogModel)
    if conversation_id:
        query = query.filter(RuntimeLogModel.conversation_id == conversation_id)
    if outline_id:
        query = query.filter(RuntimeLogModel.outline_id == outline_id)
    return query.order_by(RuntimeLogModel.created_at.desc()).limit(limit).all()

