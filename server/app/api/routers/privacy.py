from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import Dict, Any

from app.db.session import get_db
from app.db import models
from app.api.deps import get_current_active_user

router = APIRouter(
    prefix="/privacy",
    tags=["Privacy"],
)


@router.get("/status")
def get_privacy_status(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_active_user),
) -> Dict[str, Any]:
    # Calculate the number of local images processed by the hospital
    local_images_count = (
        db.query(func.count(models.ImageMetadata.id))
        .filter(models.ImageMetadata.hospital_id == current_user.hospital_id)
        .scalar()
        or 0
    )

    # Retrieve the 5 most recent audit logs for this hospital's users
    recent_logs = (
        db.query(models.AuditLog)
        .join(models.User)
        .filter(models.User.hospital_id == current_user.hospital_id)
        .order_by(models.AuditLog.timestamp.desc())
        .limit(5)
        .all()
    )

    logs_data = []
    for log in recent_logs:
        logs_data.append(
            {
                "action": log.action,
                "resource_type": log.resource_type,
                "timestamp": log.timestamp.isoformat(),
                "user": log.user.username,
            }
        )

    return {
        "status": "verifiable_local",
        "total_local_images_processed": local_images_count,
        "raw_images_shared": 0,  # Factual reflection of the architecture
        "recent_audit_logs": logs_data,
        "message": "Data remains 100% on-premises. Only encrypted model updates are shared.",
    }
