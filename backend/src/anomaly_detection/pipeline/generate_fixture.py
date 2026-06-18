"""Generate a synthetic NSL-KDD style fixture for tests and CI.

Creates ~3,000 rows with realistic NSL-KDD column distributions.
NOT real network data — synthetic, deterministic (seeded).

Run: python -m anomaly_detection.pipeline.generate_fixture
"""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Any

random.seed(42)

PROTOCOL_MAP = {"icmp": 0.0, "tcp": 1.0, "udp": 2.0}
SERVICE_MAP = {
    "http": 21,
    "ftp": 16,
    "ftp_data": 17,
    "smtp": 52,
    "ssh": 54,
    "telnet": 58,
    "private": 47,
    "domain_u": 9,
    "other": 42,
    "eco_i": 11,
}
FLAG_MAP = {"SF": 9, "S0": 5, "REJ": 1, "RSTO": 2, "SH": 10}

ATTACK_TYPES = ["BENIGN", "DoS", "Probe", "R2L", "U2R"]
ATTACK_WEIGHTS = [0.67, 0.18, 0.08, 0.05, 0.02]

NSL_COLUMNS = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
    "label",
]


def _row(
    duration: int,
    protocol: str,
    service: str,
    flag: str,
    src_b: int,
    dst_b: int,
    logged_in: int,
    count: int,
    srv_count: int,
    serr: float,
    rerr: float,
    same_srv: float,
    diff_srv: float,
    dst_cnt: int,
    dst_srv_cnt: int,
    dst_same_srv: float,
    label: str,
) -> dict[str, Any]:
    return {
        "duration": duration,
        "protocol_type": PROTOCOL_MAP.get(protocol, 1.0),
        "service": SERVICE_MAP.get(service, 42),
        "flag": FLAG_MAP.get(flag, 9),
        "src_bytes": src_b,
        "dst_bytes": dst_b,
        "land": 0,
        "wrong_fragment": random.choice([0, 0, 0, 1]),
        "urgent": 0,
        "hot": random.randint(0, 5),
        "num_failed_logins": 0,
        "logged_in": logged_in,
        "num_compromised": 0,
        "root_shell": 0,
        "su_attempted": 0,
        "num_root": 0,
        "num_file_creations": 0,
        "num_shells": 0,
        "num_access_files": 0,
        "num_outbound_cmds": 0,
        "is_host_login": 0,
        "is_guest_login": 0,
        "count": count,
        "srv_count": srv_count,
        "serror_rate": serr,
        "srv_serror_rate": serr,
        "rerror_rate": rerr,
        "srv_rerror_rate": rerr,
        "same_srv_rate": same_srv,
        "diff_srv_rate": diff_srv,
        "srv_diff_host_rate": round(random.uniform(0, 0.1), 3),
        "dst_host_count": dst_cnt,
        "dst_host_srv_count": dst_srv_cnt,
        "dst_host_same_srv_rate": dst_same_srv,
        "dst_host_diff_srv_rate": round(1 - dst_same_srv, 3),
        "dst_host_same_src_port_rate": round(random.uniform(0, 1), 3),
        "dst_host_srv_diff_host_rate": round(random.uniform(0, 0.1), 3),
        "dst_host_serror_rate": serr,
        "dst_host_srv_serror_rate": serr,
        "dst_host_rerror_rate": rerr,
        "dst_host_srv_rerror_rate": rerr,
        "label": label,
    }


def benign() -> dict[str, Any]:
    dur = random.randint(0, 200)
    return _row(
        duration=dur,
        protocol="tcp",
        service="http",
        flag="SF",
        src_b=random.randint(0, 10000),
        dst_b=random.randint(0, 50000),
        logged_in=1,
        count=random.randint(1, 100),
        srv_count=random.randint(1, 80),
        serr=0.0,
        rerr=0.0,
        same_srv=round(random.uniform(0.7, 1.0), 3),
        diff_srv=round(random.uniform(0.0, 0.3), 3),
        dst_cnt=random.randint(1, 255),
        dst_srv_cnt=random.randint(1, 200),
        dst_same_srv=round(random.uniform(0.5, 1.0), 3),
        label="BENIGN",
    )


def dos() -> dict[str, Any]:
    return _row(
        duration=random.randint(0, 5),
        protocol="tcp",
        service="http",
        flag="S0",
        src_b=random.randint(0, 500),
        dst_b=0,
        logged_in=0,
        count=random.randint(300, 511),
        srv_count=random.randint(200, 511),
        serr=round(random.uniform(0.8, 1.0), 3),
        rerr=0.0,
        same_srv=round(random.uniform(0.9, 1.0), 3),
        diff_srv=round(random.uniform(0.0, 0.1), 3),
        dst_cnt=255,
        dst_srv_cnt=255,
        dst_same_srv=round(random.uniform(0.9, 1.0), 3),
        label="DoS",
    )


def probe() -> dict[str, Any]:
    services = list(SERVICE_MAP.keys())
    return _row(
        duration=random.randint(0, 2),
        protocol=random.choice(["tcp", "udp"]),
        service=random.choice(services),
        flag=random.choice(["SF", "S0", "REJ"]),
        src_b=random.randint(0, 1000),
        dst_b=random.randint(0, 500),
        logged_in=0,
        count=random.randint(1, 50),
        srv_count=random.randint(1, 20),
        serr=round(random.uniform(0.0, 0.5), 3),
        rerr=round(random.uniform(0.0, 0.5), 3),
        same_srv=round(random.uniform(0.0, 0.3), 3),
        diff_srv=round(random.uniform(0.5, 1.0), 3),
        dst_cnt=random.randint(100, 255),
        dst_srv_cnt=random.randint(50, 255),
        dst_same_srv=round(random.uniform(0.0, 0.3), 3),
        label="Probe",
    )


def r2l() -> dict[str, Any]:
    return _row(
        duration=random.randint(1, 100),
        protocol="tcp",
        service=random.choice(["ftp", "telnet", "smtp"]),
        flag="SF",
        src_b=random.randint(100, 5000),
        dst_b=random.randint(100, 5000),
        logged_in=random.choice([0, 1]),
        count=random.randint(1, 10),
        srv_count=random.randint(1, 10),
        serr=0.0,
        rerr=0.0,
        same_srv=round(random.uniform(0.5, 1.0), 3),
        diff_srv=round(random.uniform(0.0, 0.5), 3),
        dst_cnt=random.randint(1, 50),
        dst_srv_cnt=random.randint(1, 50),
        dst_same_srv=round(random.uniform(0.5, 1.0), 3),
        label="R2L",
    )


def u2r() -> dict[str, Any]:
    row = _row(
        duration=random.randint(0, 10),
        protocol="tcp",
        service="telnet",
        flag="SF",
        src_b=random.randint(0, 2000),
        dst_b=random.randint(0, 2000),
        logged_in=1,
        count=random.randint(1, 5),
        srv_count=random.randint(1, 5),
        serr=0.0,
        rerr=0.0,
        same_srv=round(random.uniform(0.5, 1.0), 3),
        diff_srv=round(random.uniform(0.0, 0.5), 3),
        dst_cnt=random.randint(1, 20),
        dst_srv_cnt=random.randint(1, 20),
        dst_same_srv=round(random.uniform(0.5, 1.0), 3),
        label="U2R",
    )
    row["root_shell"] = random.choice([0, 1])
    row["num_compromised"] = random.randint(0, 10)
    return row


GENERATORS = {
    "BENIGN": benign,
    "DoS": dos,
    "Probe": probe,
    "R2L": r2l,
    "U2R": u2r,
}


def generate_fixture(output_path: Path, num_rows: int = 3000) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = list(GENERATORS.keys())
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=NSL_COLUMNS)
        writer.writeheader()
        for _ in range(num_rows):
            label = random.choices(labels, weights=ATTACK_WEIGHTS, k=1)[0]
            writer.writerow(GENERATORS[label]())


if __name__ == "__main__":
    fixture_path = (
        Path(__file__).resolve().parents[4] / "data" / "fixtures" / "nsl_kdd_sample.csv"
    )
    generate_fixture(fixture_path)
    print(f"Generated NSL-KDD fixture at {fixture_path}")
