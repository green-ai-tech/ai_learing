"""教学大纲业务服务。"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from sqlalchemy.orm import Session

from backend.models.file_record import FileRecordModel
from backend.models.outline import OutlineModel
from backend.schemas.outline import OutlineGenerateRequest
from backend.services.outline_agent_service import OutlineAgentRuntime
from backend.services.outline_export_service import OutlineExportService
from backend.services.pptx_generation_service import PptxGenerationService
from backend.services.runtime_service import (
    ensure_conversation,
    log_runtime_event,
    save_checkpoint,
    save_store_record,
)


XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
PPTX_MIME = "application/vnd.openxmlformats-officedocument.presentationml.presentation"


class OutlineService:
    """Outline Agent 编排、持久化与文件生成。"""

    def __init__(self):
        self.agent_runtime = OutlineAgentRuntime()
        self.export_service = OutlineExportService()
        self.pptx_service = PptxGenerationService()

    def generate_outline(self, db: Session, payload: OutlineGenerateRequest) -> OutlineModel:
        conversation_id = payload.ensure_conversation_id()
        request_data = payload.model_dump()
        request_data["conversation_id"] = conversation_id

        ensure_conversation(
            db,
            conversation_id,
            title=payload.course_title,
            metadata={"agent": "outline_agent"},
        )
        outline = OutlineModel(
            conversation_id=conversation_id,
            user_input=request_data,
            status="running",
        )
        db.add(outline)
        db.flush()

        try:
            log_runtime_event(
                db,
                event="outline_generation_started",
                message="开始生成教学大纲",
                conversation_id=conversation_id,
                outline_id=outline.id,
                payload={"course_title": payload.course_title},
            )
            save_checkpoint(
                db,
                conversation_id=conversation_id,
                checkpoint_name=f"outline_{outline.id}_request_received",
                state={"user_input": request_data, "status": "running"},
            )
            db.commit()

            outline_json, source = self.agent_runtime.generate(payload)

            outline.outline_json = outline_json
            outline.status = "completed"
            db.add(outline)
            log_runtime_event(
                db,
                event="outline_agent_completed",
                message="LLM 调用完成" if source == "llm" else "本地结构化大纲生成完成",
                conversation_id=conversation_id,
                outline_id=outline.id,
                payload={"source": source},
            )
            save_store_record(
                db,
                conversation_id=conversation_id,
                namespace="outline",
                record_key=f"outline:{outline.id}",
                value=outline_json,
            )
            save_checkpoint(
                db,
                conversation_id=conversation_id,
                checkpoint_name=f"outline_{outline.id}_completed",
                state={"outline_json": outline_json, "status": "completed"},
            )
            log_runtime_event(
                db,
                event="outline_saved",
                message="大纲入库完成",
                conversation_id=conversation_id,
                outline_id=outline.id,
            )
            db.commit()
            db.refresh(outline)
            return outline
        except Exception as exc:
            db.rollback()
            failed = db.get(OutlineModel, outline.id)
            if failed:
                failed.status = "failed"
                failed.error_message = str(exc)
                db.add(failed)
                log_runtime_event(
                    db,
                    event="outline_generation_failed",
                    message=f"大纲生成失败: {exc}",
                    conversation_id=conversation_id,
                    outline_id=outline.id,
                    level="ERROR",
                )
                db.commit()
            raise

    def list_outlines(
        self,
        db: Session,
        *,
        conversation_id: Optional[str] = None,
        limit: int = 50,
    ) -> list[OutlineModel]:
        query = db.query(OutlineModel)
        if conversation_id:
            query = query.filter(OutlineModel.conversation_id == conversation_id)
        return query.order_by(OutlineModel.created_at.desc()).limit(limit).all()

    def get_outline(self, db: Session, outline_id: int) -> Optional[OutlineModel]:
        return db.get(OutlineModel, outline_id)

    def generate_xlsx(self, db: Session, outline: OutlineModel) -> Path:
        file_path = self.export_service.generate_xlsx(outline)
        outline.xlsx_file_path = str(file_path)
        db.add(outline)
        self._record_file(db, outline, file_path, "xlsx", XLSX_MIME)
        log_runtime_event(
            db,
            event="xlsx_generated",
            message="XLSX 生成完成",
            conversation_id=outline.conversation_id,
            outline_id=outline.id,
            payload={"file_path": str(file_path)},
        )
        db.commit()
        db.refresh(outline)
        return file_path

    def generate_pptx(self, db: Session, outline: OutlineModel) -> Path:
        file_path = self.pptx_service.generate_pptx(outline)
        outline.pptx_file_path = str(file_path)
        db.add(outline)
        self._record_file(db, outline, file_path, "pptx", PPTX_MIME)
        log_runtime_event(
            db,
            event="pptx_generated",
            message="PPTX 生成完成",
            conversation_id=outline.conversation_id,
            outline_id=outline.id,
            payload={"file_path": str(file_path)},
        )
        db.commit()
        db.refresh(outline)
        return file_path

    def _record_file(
        self,
        db: Session,
        outline: OutlineModel,
        file_path: Path,
        file_type: str,
        mime_type: str,
    ) -> FileRecordModel:
        record = FileRecordModel(
            outline_id=outline.id,
            conversation_id=outline.conversation_id,
            file_type=file_type,
            file_name=file_path.name,
            file_path=str(file_path),
            mime_type=mime_type,
        )
        db.add(record)
        db.flush()
        return record
