"""Attacks router — detect, list, and block attacking IPs."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse
from sqlalchemy import func, select
from starlette.responses import Response

from anomaly_detection.db.models import AlertSeverity, Attack, BlockedIP
from anomaly_detection.schemas.common import AttackResponse, BlockIPRequest

router = APIRouter(prefix="/api/v1/attacks", tags=["attacks"])


@router.get("")
async def list_attacks(
    request: Request,
    page: int = Query(1, ge=1),
    per_page: int = Query(25, ge=1, le=100),
    attack_type: str | None = None,
    severity: str | None = None,
    search: str | None = None,
) -> dict[str, Any]:
    """List detected attacks with pagination and filtering."""
    session_factory = request.app.state.session_factory

    async with session_factory() as session:
        query = select(Attack)
        count_query = select(func.count(Attack.id))

        if attack_type:
            query = query.where(Attack.attack_type == attack_type)
            count_query = count_query.where(Attack.attack_type == attack_type)
        if severity:
            query = query.where(Attack.severity == AlertSeverity(severity))
            count_query = count_query.where(Attack.severity == AlertSeverity(severity))
        if search:
            search_filter = Attack.src_ip.ilike(f"%{search}%") | Attack.dst_ip.ilike(f"%{search}%")
            query = query.where(search_filter)
            count_query = count_query.where(search_filter)

        total = (await session.execute(count_query)).scalar() or 0
        offset = (page - 1) * per_page
        result = await session.execute(
            query.order_by(Attack.detected_at.desc()).offset(offset).limit(per_page)
        )
        attacks = result.scalars().all()

        return {
            "items": [
                AttackResponse(
                    id=a.id,
                    attack_type=a.attack_type,
                    severity=a.severity.value,
                    confidence=a.confidence,
                    src_ip=a.src_ip,
                    dst_ip=a.dst_ip,
                    src_port=a.src_port,
                    dst_port=a.dst_port,
                    protocol=a.protocol,
                    recommendation=a.recommendation,
                    is_blocked=a.is_blocked,
                    detected_at=a.detected_at,
                ).model_dump()
                for a in attacks
            ],
            "total": total,
            "page": page,
            "per_page": per_page,
            "total_pages": max(1, (total + per_page - 1) // per_page),
        }


@router.get("/{attack_id}")
async def get_attack(request: Request, attack_id: str) -> Response | dict[str, Any]:
    """Get attack details."""
    session_factory = request.app.state.session_factory

    async with session_factory() as session:
        result = await session.execute(select(Attack).where(Attack.id == attack_id))
        attack = result.scalar_one_or_none()
        if not attack:
            return JSONResponse(status_code=404, content={"detail": "Attack not found"})

        return AttackResponse(
            id=attack.id,
            attack_type=attack.attack_type,
            severity=attack.severity.value,
            confidence=attack.confidence,
            src_ip=attack.src_ip,
            dst_ip=attack.dst_ip,
            src_port=attack.src_port,
            dst_port=attack.dst_port,
            protocol=attack.protocol,
            recommendation=attack.recommendation,
            is_blocked=attack.is_blocked,
            detected_at=attack.detected_at,
        ).model_dump()


@router.post("/block-ip")
async def block_ip(request: Request, body: BlockIPRequest) -> dict[str, Any]:
    """Block an attacking IP address."""
    session_factory = request.app.state.session_factory

    async with session_factory() as session:
        # Check if already blocked
        result = await session.execute(
            select(BlockedIP).where(BlockedIP.ip_address == body.ip_address)
        )
        if result.scalar_one_or_none():
            return {"message": f"IP {body.ip_address} is already blocked"}

        blocked = BlockedIP(
            ip_address=body.ip_address,
            reason=body.reason or "Blocked from attack detection",
            attack_type=body.attack_type,
        )
        session.add(blocked)

        # Mark related attacks as blocked
        result = await session.execute(select(Attack).where(Attack.src_ip == body.ip_address))
        for attack in result.scalars().all():
            attack.is_blocked = True

        await session.commit()

    return {"message": f"IP {body.ip_address} blocked successfully"}


@router.get("/blocked-ips/list")
async def list_blocked_ips(request: Request) -> list[dict[str, Any]]:
    """List all blocked IP addresses."""
    session_factory = request.app.state.session_factory

    async with session_factory() as session:
        result = await session.execute(select(BlockedIP).order_by(BlockedIP.blocked_at.desc()))
        ips = result.scalars().all()
        return [
            {
                "id": str(ip.id),
                "ip_address": ip.ip_address,
                "reason": ip.reason,
                "attack_type": ip.attack_type,
                "blocked_at": ip.blocked_at.isoformat(),
            }
            for ip in ips
        ]


@router.delete("/blocked-ips/{ip_address}")
async def unblock_ip(request: Request, ip_address: str) -> Response | dict[str, Any]:
    """Remove an IP from the blocked list."""
    session_factory = request.app.state.session_factory

    async with session_factory() as session:
        result = await session.execute(select(BlockedIP).where(BlockedIP.ip_address == ip_address))
        blocked = result.scalar_one_or_none()
        if blocked:
            await session.delete(blocked)
            await session.commit()
            return {"message": f"IP {ip_address} unblocked"}
        return JSONResponse(status_code=404, content={"detail": "IP not found in blocked list"})
