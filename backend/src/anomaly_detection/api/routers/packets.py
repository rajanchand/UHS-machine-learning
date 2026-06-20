"""Packets router — capture, list, filter, export packets."""

from __future__ import annotations

import csv
import io
import random
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Query, Request
from fastapi.responses import StreamingResponse
from sqlalchemy import func, select

from anomaly_detection.db.models import CaptureStatus, Packet, PacketCapture
from anomaly_detection.schemas.common import PacketCaptureResponse, PacketResponse

router = APIRouter(prefix="/api/v1/packets", tags=["packets"])


@router.post("/capture/start", response_model=PacketCaptureResponse)
async def start_capture(request: Request, body: dict[str, Any] | None = None) -> PacketCaptureResponse:
    """Start a packet capture session."""
    session_factory = request.app.state.session_factory
    interface = (body or {}).get("interface", "eth0")

    async with session_factory() as session:
        capture = PacketCapture(
            interface=interface,
            status=CaptureStatus.RUNNING,
            packet_count=0,
        )
        session.add(capture)
        await session.commit()
        await session.refresh(capture)

        # Generate some simulated packets for the capture
        protocols = ["TCP", "UDP", "ICMP", "HTTP", "HTTPS", "DNS"]
        for _ in range(random.randint(20, 50)):
            pkt = Packet(
                capture_id=capture.id,
                src_ip=f"192.168.{random.randint(1, 10)}.{random.randint(1, 254)}",
                dst_ip=f"10.0.{random.randint(0, 5)}.{random.randint(1, 254)}",
                protocol=random.choice(protocols),
                src_port=random.randint(1024, 65535),
                dst_port=random.choice([80, 443, 22, 53, 8080, 3306]),
                packet_size=random.randint(40, 1500),
                ttl=random.choice([64, 128, 255]),
                flags=random.choice(["SYN", "ACK", "SYN-ACK", "FIN", "RST", "PSH-ACK", ""]),
                status=random.choices(["Normal", "Suspicious", "Malicious"], weights=[85, 10, 5])[
                    0
                ],
            )
            session.add(pkt)

        capture.packet_count = random.randint(20, 50)
        await session.commit()
        await session.refresh(capture)

        return PacketCaptureResponse(
            id=capture.id,
            interface=capture.interface,
            status=capture.status.value,
            packet_count=capture.packet_count,
            started_at=capture.started_at,
            stopped_at=capture.stopped_at,
        )


@router.post("/capture/stop")
async def stop_capture(request: Request, body: dict[str, Any] | None = None) -> dict[str, Any]:
    """Stop a running packet capture session."""
    session_factory = request.app.state.session_factory
    capture_id = (body or {}).get("capture_id")

    async with session_factory() as session:
        if capture_id:
            result = await session.execute(
                select(PacketCapture).where(PacketCapture.id == capture_id)
            )
            capture = result.scalar_one_or_none()
            if capture:
                capture.status = CaptureStatus.COMPLETED
                capture.stopped_at = datetime.now(UTC)
                await session.commit()
        else:
            # Stop all running captures
            result = await session.execute(
                select(PacketCapture).where(PacketCapture.status == CaptureStatus.RUNNING)
            )
            captures = result.scalars().all()
            for c in captures:
                c.status = CaptureStatus.COMPLETED
                c.stopped_at = datetime.now(UTC)
            await session.commit()

    return {"message": "Capture stopped"}


@router.get("")
async def list_packets(
    request: Request,
    page: int = Query(1, ge=1),
    per_page: int = Query(25, ge=1, le=100),
    protocol: str | None = None,
    status: str | None = None,
    search: str | None = None,
) -> dict[str, Any]:
    """List captured packets with pagination, filtering, and search."""
    session_factory = request.app.state.session_factory

    async with session_factory() as session:
        query = select(Packet)
        count_query = select(func.count(Packet.id))

        if protocol:
            query = query.where(Packet.protocol == protocol)
            count_query = count_query.where(Packet.protocol == protocol)
        if status:
            query = query.where(Packet.status == status)
            count_query = count_query.where(Packet.status == status)
        if search:
            search_filter = Packet.src_ip.ilike(f"%{search}%") | Packet.dst_ip.ilike(f"%{search}%")
            query = query.where(search_filter)
            count_query = count_query.where(search_filter)

        total = (await session.execute(count_query)).scalar() or 0
        offset = (page - 1) * per_page
        result = await session.execute(
            query.order_by(Packet.timestamp.desc()).offset(offset).limit(per_page)
        )
        packets = result.scalars().all()

        return {
            "items": [
                PacketResponse(
                    id=p.id,
                    timestamp=p.timestamp,
                    src_ip=p.src_ip,
                    dst_ip=p.dst_ip,
                    protocol=p.protocol,
                    src_port=p.src_port,
                    dst_port=p.dst_port,
                    packet_size=p.packet_size,
                    ttl=p.ttl,
                    flags=p.flags,
                    status=p.status,
                ).model_dump()
                for p in packets
            ],
            "total": total,
            "page": page,
            "per_page": per_page,
            "total_pages": max(1, (total + per_page - 1) // per_page),
        }


@router.get("/export/csv")
async def export_packets_csv(request: Request) -> StreamingResponse:
    """Export all packets as CSV file."""
    session_factory = request.app.state.session_factory

    async with session_factory() as session:
        result = await session.execute(
            select(Packet).order_by(Packet.timestamp.desc()).limit(10000)
        )
        packets = result.scalars().all()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(
        [
            "ID",
            "Timestamp",
            "Source IP",
            "Destination IP",
            "Protocol",
            "Source Port",
            "Destination Port",
            "Packet Size",
            "TTL",
            "Flags",
            "Status",
        ]
    )
    for p in packets:
        writer.writerow(
            [
                str(p.id),
                p.timestamp.isoformat(),
                p.src_ip,
                p.dst_ip,
                p.protocol,
                p.src_port,
                p.dst_port,
                p.packet_size,
                p.ttl,
                p.flags,
                p.status,
            ]
        )

    output.seek(0)
    return StreamingResponse(
        io.BytesIO(output.getvalue().encode()),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=packets_export.csv"},
    )
