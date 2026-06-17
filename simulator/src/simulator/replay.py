"""Streaming traffic simulator — replays benchmark flows to the API.

Reads from a processed parquet file or fixture CSV and POSTs flows to
the API at a configurable rate. Assigns current timestamps to simulate
a live traffic stream.

Usage:
    python -m simulator.replay
"""

from __future__ import annotations

import asyncio
import random
import signal
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pandas as pd
import structlog

from typing import Any

from simulator.config import get_simulator_settings

structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(colors=True),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    logger_factory=structlog.stdlib.LoggerFactory(),
)

logger = structlog.get_logger(__name__)

# Feature column mapping from CICIDS2017 to our internal names
FEATURE_MAP: dict[str, str] = {
    "Flow Duration": "flow_duration",
    "Total Fwd Packets": "total_fwd_packets",
    "Total Backward Packets": "total_bwd_packets",
    "Total Length of Fwd Packets": "total_len_fwd_packets",
    "Total Length of Bwd Packets": "total_len_bwd_packets",
    "Fwd Packet Length Max": "fwd_packet_len_max",
    "Fwd Packet Length Min": "fwd_packet_len_min",
    "Fwd Packet Length Mean": "fwd_packet_len_mean",
    "Fwd Packet Length Std": "fwd_packet_len_std",
    "Bwd Packet Length Max": "bwd_packet_len_max",
    "Bwd Packet Length Min": "bwd_packet_len_min",
    "Bwd Packet Length Mean": "bwd_packet_len_mean",
    "Bwd Packet Length Std": "bwd_packet_len_std",
    "Flow Bytes/s": "flow_bytes_per_s",
    "Flow Packets/s": "flow_packets_per_s",
    "FIN Flag Count": "fin_flag_count",
    "SYN Flag Count": "syn_flag_count",
    "RST Flag Count": "rst_flag_count",
    "PSH Flag Count": "psh_flag_count",
    "ACK Flag Count": "ack_flag_count",
    "URG Flag Count": "urg_flag_count",
    "Flow IAT Mean": "flow_iat_mean",
    "Flow IAT Std": "flow_iat_std",
    "Flow IAT Max": "flow_iat_max",
    "Flow IAT Min": "flow_iat_min",
    "Fwd IAT Mean": "fwd_iat_mean",
    "Fwd IAT Std": "fwd_iat_std",
    "Fwd IAT Max": "fwd_iat_max",
    "Fwd IAT Min": "fwd_iat_min",
    "Bwd IAT Mean": "bwd_iat_mean",
    "Bwd IAT Std": "bwd_iat_std",
    "Bwd IAT Max": "bwd_iat_max",
    "Bwd IAT Min": "bwd_iat_min",
    "Down/Up Ratio": "down_up_ratio",
    "Average Packet Size": "avg_packet_size",
    "Avg Fwd Segment Size": "avg_fwd_segment_size",
    "Avg Bwd Segment Size": "avg_bwd_segment_size",
}

INTERNAL_FEATURE_NAMES = list(FEATURE_MAP.values())

# Shutdown flag
_shutdown = False


def _handle_signal(signum: int, frame: object) -> None:
    """Handle graceful shutdown."""
    global _shutdown
    _shutdown = True
    logger.info("shutdown_requested", signal=signum)


def load_data(data_path: str) -> pd.DataFrame:
    """Load flow data from CSV or parquet."""
    path = Path(data_path)

    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix == ".csv":
        df = pd.read_csv(path, low_memory=False)
        # Strip column names (CICIDS2017 quirk)
        df.columns = df.columns.str.strip()
        # Rename to internal names
        rename_map = {k: v for k, v in FEATURE_MAP.items() if k in df.columns}
        df = df.rename(columns=rename_map)
        # Also rename ID columns
        id_renames = {
            "Source IP": "src_ip",
            "Destination IP": "dst_ip",
            "Source Port": "src_port",
            "Destination Port": "dst_port",
            "Protocol": "protocol",
            "Label": "label",
        }
        df = df.rename(columns={k: v for k, v in id_renames.items() if k in df.columns})
    else:
        msg = f"Unsupported file format: {path.suffix}"
        raise ValueError(msg)

    logger.info("data_loaded", path=str(path), rows=len(df))
    return df


def _randomize_ip(ip: str) -> str:
    """Slightly randomize the last octet of an IP address."""
    parts = ip.split(".")
    if len(parts) == 4:
        parts[-1] = str(random.randint(1, 254))
    return ".".join(parts)


def row_to_payload(row: pd.Series) -> dict[str, Any]:
    """Convert a DataFrame row to an API payload."""
    now = datetime.now(timezone.utc).isoformat()

    # Extract features
    features: dict[str, float] = {}
    for feat_name in INTERNAL_FEATURE_NAMES:
        if feat_name in row.index:
            val = row[feat_name]
            try:
                features[feat_name] = max(0.0, float(val)) if pd.notna(val) else 0.0
            except (ValueError, TypeError):
                features[feat_name] = 0.0
        else:
            features[feat_name] = 0.0

    src_ip = str(row.get("src_ip", "192.168.1.1"))
    dst_ip = str(row.get("dst_ip", "10.0.0.1"))

    return {
        "ts": now,
        "src_ip": _randomize_ip(src_ip),
        "src_port": int(row.get("src_port", random.randint(1024, 65535))),
        "dst_ip": _randomize_ip(dst_ip),
        "dst_port": int(row.get("dst_port", 80)),
        "protocol": int(row.get("protocol", 6)),
        "features": features,
        "label": str(row.get("label", "UNKNOWN")) if pd.notna(row.get("label")) else None,
    }


async def replay_flows(
    df: pd.DataFrame,
    api_url: str,
    rate_per_sec: float,
) -> None:
    """Replay flows from DataFrame as HTTP requests.

    Args:
        df: DataFrame with flow data.
        api_url: Base URL of the API.
        rate_per_sec: Number of flows to send per second.
    """
    endpoint = f"{api_url}/api/v1/flows/stream"
    interval = 1.0 / rate_per_sec if rate_per_sec > 0 else 1.0

    logger.info(
        "starting_replay",
        endpoint=endpoint,
        rate=rate_per_sec,
        total_flows=len(df),
    )

    # Pre-slice scenario datasets to speed up simulation injection
    portscan_rows = df[df["label"].str.upper() == "PORTSCAN"] if "label" in df.columns else pd.DataFrame()
    ddos_rows = df[df["label"].str.upper() == "DDOS"] if "label" in df.columns else pd.DataFrame()
    bruteforce_rows = df[df["label"].str.upper().isin(["FTP-PATATOR", "SSH-PATATOR", "WEB ATTACK - BRUTE FORCE"])] if "label" in df.columns else pd.DataFrame()

    sent = 0
    errors = 0
    scenario_count = 0

    headers = {"X-API-Key": "simulator-secret"}

    async with httpx.AsyncClient(timeout=10.0, headers=headers) as client:
        # Wait for API to be ready
        for attempt in range(30):
            if _shutdown:
                return
            try:
                resp = await client.get(f"{api_url}/health")
                if resp.status_code == 200:
                    logger.info("api_ready")
                    break
            except httpx.ConnectError:
                pass
            logger.info("waiting_for_api", attempt=attempt + 1)
            await asyncio.sleep(2)
        else:
            logger.error("api_not_ready_after_60s")
            return

        # Replay loop
        while not _shutdown:
            for idx in range(len(df)):
                if _shutdown:
                    break

                # 1. Poll backend for active simulation scenario
                active_scenario = None
                try:
                    resp = await client.get(f"{api_url}/simulate")
                    if resp.status_code == 200:
                        active_scenario = resp.json().get("active_scenario")
                except Exception as e:
                    logger.warning("failed_to_poll_scenario", error=str(e))

                # 2. Select flow based on active scenario or standard replay
                row = None
                if active_scenario == "port_scan" and len(portscan_rows) > 0:
                    row = portscan_rows.sample(1).iloc[0]
                    scenario_count += 1
                elif active_scenario == "ddos" and len(ddos_rows) > 0:
                    row = ddos_rows.sample(1).iloc[0]
                    scenario_count += 1
                elif active_scenario == "brute_force" and len(bruteforce_rows) > 0:
                    row = bruteforce_rows.sample(1).iloc[0]
                    scenario_count += 1
                else:
                    # Fallback to benign or default sequential flow replay
                    row = df.iloc[idx]
                    scenario_count = 0

                payload = row_to_payload(row)

                try:
                    response = await client.post(endpoint, json=payload)
                    if response.status_code == 200:
                        sent += 1
                    else:
                        errors += 1
                        if errors <= 5:
                            logger.warning(
                                "api_error",
                                status=response.status_code,
                                body=response.text[:200],
                            )
                except httpx.ConnectError:
                    errors += 1
                    if errors % 10 == 1:
                        logger.warning("connection_error", total_errors=errors)
                    await asyncio.sleep(5)
                    continue

                # 3. If scenario burst reaches limit, reset scenario on backend
                if active_scenario and scenario_count >= 20:
                    logger.info("scenario_burst_complete", scenario=active_scenario, count=scenario_count)
                    try:
                        await client.post(f"{api_url}/simulate", json={"scenario": None})
                    except Exception as e:
                        logger.warning("failed_to_reset_scenario", error=str(e))
                    scenario_count = 0

                if sent % 100 == 0:
                    logger.info("progress", sent=sent, errors=errors)

                await asyncio.sleep(interval)

            logger.info("replay_cycle_complete", sent=sent, errors=errors)
            # Loop back to beginning for continuous replay

    logger.info("replay_stopped", total_sent=sent, total_errors=errors)


async def main() -> None:
    """Main entrypoint for the simulator."""
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    settings = get_simulator_settings()
    logger.info(
        "simulator_starting",
        api_url=settings.simulator_api_url,
        rate=settings.simulator_rate_per_sec,
        data_path=settings.simulator_data_path,
    )

    df = load_data(settings.simulator_data_path)
    await replay_flows(
        df=df,
        api_url=settings.simulator_api_url,
        rate_per_sec=settings.simulator_rate_per_sec,
    )


if __name__ == "__main__":
    asyncio.run(main())
