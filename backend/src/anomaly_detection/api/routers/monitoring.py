"""Monitoring router — live network monitoring, interface listing, SSE feed."""

from __future__ import annotations

import asyncio
import json
import random
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

router = APIRouter(prefix="/api/v1/monitoring", tags=["monitoring"])

# In-memory monitoring state
_monitoring_state: dict[str, Any] = {
    "is_running": False,
    "interface": "eth0",
    "packet_count": 0,
    "incoming_bytes": 0,
    "outgoing_bytes": 0,
    "start_time": None,
}


@router.post("/start")
async def start_monitoring(request: Request, body: dict[str, Any] | None = None) -> dict[str, Any]:
    """Start live network monitoring."""
    interface = (body or {}).get("interface", "eth0")
    _monitoring_state["is_running"] = True
    _monitoring_state["interface"] = interface
    _monitoring_state["packet_count"] = 0
    _monitoring_state["incoming_bytes"] = 0
    _monitoring_state["outgoing_bytes"] = 0
    _monitoring_state["start_time"] = datetime.now(UTC).isoformat()
    return {"message": "Monitoring started", "interface": interface}


@router.post("/stop")
async def stop_monitoring() -> dict[str, Any]:
    """Stop live network monitoring."""
    _monitoring_state["is_running"] = False
    return {"message": "Monitoring stopped", "total_packets": _monitoring_state["packet_count"]}


@router.get("/status")
async def monitoring_status() -> dict[str, Any]:
    """Get current monitoring state."""
    return {
        "is_running": _monitoring_state["is_running"],
        "interface": _monitoring_state["interface"],
        "packet_count": _monitoring_state["packet_count"],
        "incoming_bytes": _monitoring_state["incoming_bytes"],
        "outgoing_bytes": _monitoring_state["outgoing_bytes"],
        "start_time": _monitoring_state["start_time"],
    }


@router.get("/interfaces")
async def list_interfaces() -> list[dict[str, Any]]:
    """List available network interfaces."""
    # Return common interface names for demo
    interfaces = [
        {"name": "eth0", "description": "Ethernet", "status": "up"},
        {"name": "wlan0", "description": "Wi-Fi", "status": "up"},
        {"name": "lo", "description": "Loopback", "status": "up"},
        {"name": "docker0", "description": "Docker Bridge", "status": "down"},
    ]
    try:
        import psutil

        real_interfaces = psutil.net_if_addrs()
        interfaces = [
            {"name": name, "description": name, "status": "up"} for name in real_interfaces
        ]
    except Exception:
        pass
    return interfaces


@router.get("/stats")
async def monitoring_stats() -> dict[str, Any]:
    """Get live monitoring statistics."""
    protocols = {
        "TCP": random.randint(500, 2000),
        "UDP": random.randint(100, 800),
        "ICMP": random.randint(10, 100),
        "HTTP": random.randint(200, 1000),
        "HTTPS": random.randint(300, 1500),
        "DNS": random.randint(50, 300),
    }
    return {
        "is_running": _monitoring_state["is_running"],
        "packet_count": _monitoring_state["packet_count"],
        "incoming_traffic": f"{random.uniform(1, 50):.1f} MB/s",
        "outgoing_traffic": f"{random.uniform(0.5, 20):.1f} MB/s",
        "bandwidth_usage": random.uniform(10, 80),
        "protocol_stats": protocols,
        "connection_status": "Connected" if _monitoring_state["is_running"] else "Disconnected",
    }


@router.get("/feed")
async def monitoring_feed(request: Request) -> StreamingResponse:
    """SSE endpoint for real-time packet stream (simulated for demo)."""

    async def event_generator() -> AsyncGenerator[str, None]:
        protocols = ["TCP", "UDP", "ICMP", "HTTP", "HTTPS", "DNS", "SSH", "FTP"]
        while True:
            if not _monitoring_state["is_running"]:
                yield 'data: {"status": "paused"}\n\n'
                await asyncio.sleep(2)
                continue

            _monitoring_state["packet_count"] = int(_monitoring_state["packet_count"]) + 1
            packet = {
                "id": _monitoring_state["packet_count"],
                "timestamp": datetime.now(UTC).isoformat(),
                "src_ip": f"192.168.{random.randint(1, 10)}.{random.randint(1, 254)}",
                "dst_ip": f"10.0.{random.randint(0, 5)}.{random.randint(1, 254)}",
                "protocol": random.choice(protocols),
                "src_port": random.randint(1024, 65535),
                "dst_port": random.choice([80, 443, 22, 53, 8080, 3306, 5432]),
                "size": random.randint(40, 1500),
                "status": random.choices(
                    ["Normal", "Suspicious", "Malicious"], weights=[85, 10, 5]
                )[0],
            }
            _monitoring_state["incoming_bytes"] = int(_monitoring_state["incoming_bytes"]) + packet["size"]

            yield f"data: {json.dumps(packet)}\n\n"
            await asyncio.sleep(random.uniform(0.1, 0.5))

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
