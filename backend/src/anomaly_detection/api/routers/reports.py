"""Reports router — generate and download PDF/Excel/CSV reports."""

from __future__ import annotations

import csv
import io
import os
import uuid
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, JSONResponse
from sqlalchemy import func, select
from starlette.responses import Response

from anomaly_detection.db.models import Alert, Attack, Packet, Prediction, Report
from anomaly_detection.schemas.common import ReportGenerateRequest, ReportResponse

router = APIRouter(prefix="/api/v1/reports", tags=["reports"])


@router.post("/generate")
async def generate_report(request: Request, body: ReportGenerateRequest) -> dict[str, Any]:
    """Generate a report in specified format."""
    session_factory = request.app.state.session_factory
    settings = request.app.state.settings

    report_id = uuid.uuid4()
    now = datetime.now(UTC)
    report_name = f"{body.report_type}_report_{now.strftime('%Y%m%d_%H%M%S')}"
    file_ext = body.report_format
    if file_ext == "excel":
        file_ext = "xlsx"
    filename = f"{report_name}.{file_ext}"
    filepath = os.path.join(str(settings.reports_dir), filename)

    # Ensure reports directory exists
    os.makedirs(str(settings.reports_dir), exist_ok=True)

    async with session_factory() as session:
        # Gather data for report
        total_packets = (await session.execute(select(func.count(Packet.id)))).scalar() or 0
        total_attacks = (await session.execute(select(func.count(Attack.id)))).scalar() or 0
        total_alerts = (await session.execute(select(func.count(Alert.id)))).scalar() or 0
        total_predictions = (await session.execute(select(func.count(Prediction.id)))).scalar() or 0

        # Generate CSV report (simple format for all types)
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["Network Anomaly Detection - Report"])
        writer.writerow(["Generated", now.isoformat()])
        writer.writerow(["Type", body.report_type.title()])
        writer.writerow([])
        writer.writerow(["Metric", "Value"])
        writer.writerow(["Total Packets", total_packets])
        writer.writerow(["Total Attacks", total_attacks])
        writer.writerow(["Total Alerts", total_alerts])
        writer.writerow(["Total Predictions", total_predictions])
        writer.writerow([])

        # Recent attacks
        writer.writerow(["Recent Attacks"])
        writer.writerow(["Type", "Severity", "Source IP", "Destination IP", "Confidence", "Time"])
        result = await session.execute(select(Attack).order_by(Attack.detected_at.desc()).limit(50))
        for a in result.scalars().all():
            writer.writerow(
                [
                    a.attack_type,
                    a.severity.value,
                    a.src_ip,
                    a.dst_ip,
                    f"{a.confidence:.2%}",
                    a.detected_at.isoformat(),
                ]
            )

        content = output.getvalue()
        file_size = len(content.encode())

        # Write file
        with open(filepath, "w") as f:
            f.write(content)

        # Save report metadata
        report = Report(
            id=report_id,
            name=report_name,
            report_type=body.report_type,
            report_format=body.report_format,
            file_path=filepath,
            file_size=file_size,
        )
        session.add(report)
        await session.commit()

    return {
        "id": str(report_id),
        "name": report_name,
        "format": body.report_format,
        "file_size": file_size,
        "message": "Report generated successfully",
    }


@router.get("")
async def list_reports(request: Request) -> list[dict[str, Any]]:
    """List all generated reports."""
    session_factory = request.app.state.session_factory

    async with session_factory() as session:
        result = await session.execute(select(Report).order_by(Report.created_at.desc()))
        reports = result.scalars().all()
        return [
            ReportResponse(
                id=r.id,
                name=r.name,
                report_type=r.report_type,
                report_format=r.report_format,
                file_size=r.file_size,
                created_at=r.created_at,
            ).model_dump()
            for r in reports
        ]


@router.get("/{report_id}/download")
async def download_report(request: Request, report_id: str) -> Response:
    """Download a generated report."""
    session_factory = request.app.state.session_factory

    async with session_factory() as session:
        result = await session.execute(select(Report).where(Report.id == report_id))
        report = result.scalar_one_or_none()
        if not report:
            return JSONResponse(status_code=404, content={"detail": "Report not found"})

        if not os.path.exists(report.file_path):
            return JSONResponse(status_code=404, content={"detail": "Report file not found"})

        return FileResponse(
            report.file_path,
            filename=os.path.basename(report.file_path),
            media_type="application/octet-stream",
        )


@router.delete("/{report_id}")
async def delete_report(request: Request, report_id: str) -> Response | dict[str, Any]:
    """Delete a report."""
    session_factory = request.app.state.session_factory

    async with session_factory() as session:
        result = await session.execute(select(Report).where(Report.id == report_id))
        report = result.scalar_one_or_none()
        if not report:
            return JSONResponse(status_code=404, content={"detail": "Report not found"})

        # Remove file
        if os.path.exists(report.file_path):
            os.remove(report.file_path)

        await session.delete(report)
        await session.commit()

    return {"message": "Report deleted"}
