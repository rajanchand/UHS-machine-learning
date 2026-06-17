"""Generate a realistic fixture dataset for tests and CI.

This creates a ~3,000 row CSV that mimics CICIDS2017 column names and
value distributions. It is NOT real network data — it is synthetic,
deterministic (seeded), and checked into the repo so tests run without
the full dataset.

Run: python -m anomaly_detection.pipeline.generate_fixture
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

# Seed for reproducibility
random.seed(42)

# CICIDS2017 column names (with the leading-space quirk on some)
COLUMNS = [
    " Destination Port",
    " Flow Duration",
    " Total Fwd Packets",
    " Total Backward Packets",
    "Total Length of Fwd Packets",
    " Total Length of Bwd Packets",
    "Fwd Packet Length Max",
    " Fwd Packet Length Min",
    " Fwd Packet Length Mean",
    " Fwd Packet Length Std",
    "Bwd Packet Length Max",
    " Bwd Packet Length Min",
    " Bwd Packet Length Mean",
    " Bwd Packet Length Std",
    "Flow Bytes/s",
    " Flow Packets/s",
    " FIN Flag Count",
    " SYN Flag Count",
    " RST Flag Count",
    " PSH Flag Count",
    " ACK Flag Count",
    " URG Flag Count",
    "Flow IAT Mean",
    " Flow IAT Std",
    " Flow IAT Max",
    " Flow IAT Min",
    "Fwd IAT Mean",
    " Fwd IAT Std",
    " Fwd IAT Max",
    " Fwd IAT Min",
    "Bwd IAT Mean",
    " Bwd IAT Std",
    " Bwd IAT Max",
    " Bwd IAT Min",
    " Down/Up Ratio",
    " Average Packet Size",
    " Avg Fwd Segment Size",
    " Avg Bwd Segment Size",
    " Label",
]

# Additional columns for flow identification
ID_COLUMNS = [
    "Flow ID",
    " Source IP",
    " Source Port",
    " Destination IP",
    " Protocol",
    " Timestamp",
]

ATTACK_TYPES = [
    "BENIGN",
    "DDoS",
    "DoS Hulk",
    "DoS GoldenEye",
    "DoS slowloris",
    "DoS Slowhttptest",
    "PortScan",
    "FTP-Patator",
    "SSH-Patator",
    "Bot",
    "Web Attack - Brute Force",
    "Web Attack - XSS",
    "Web Attack - Sql Injection",
    "Infiltration",
    "Heartbleed",
]

# Distribution: ~80% benign, rest split among attacks
ATTACK_WEIGHTS = {
    "BENIGN": 0.80,
    "DDoS": 0.05,
    "DoS Hulk": 0.04,
    "PortScan": 0.035,
    "DoS GoldenEye": 0.015,
    "DoS slowloris": 0.01,
    "DoS Slowhttptest": 0.01,
    "FTP-Patator": 0.008,
    "SSH-Patator": 0.007,
    "Bot": 0.008,
    "Web Attack - Brute Force": 0.005,
    "Web Attack - XSS": 0.003,
    "Web Attack - Sql Injection": 0.002,
    "Infiltration": 0.005,
    "Heartbleed": 0.002,
}

# Timestamp days matching CICIDS2017 collection schedule
DAYS = [
    "3/7/2017",   # Monday
    "4/7/2017",   # Tuesday
    "5/7/2017",   # Wednesday
    "6/7/2017",   # Thursday
    "7/7/2017",   # Friday
]

IPS = [
    "192.168.10.1", "192.168.10.5", "192.168.10.8", "192.168.10.12",
    "192.168.10.14", "192.168.10.15", "192.168.10.25", "192.168.10.50",
    "172.16.0.1", "172.16.0.5", "205.174.165.68", "205.174.165.73",
]


def generate_benign_row() -> dict[str, str]:
    """Generate a benign flow with typical web traffic characteristics."""
    return {
        " Destination Port": str(random.choice([80, 443, 8080, 53, 22, 25])),
        " Flow Duration": str(random.randint(0, 120000000)),
        " Total Fwd Packets": str(random.randint(1, 50)),
        " Total Backward Packets": str(random.randint(0, 40)),
        "Total Length of Fwd Packets": str(random.uniform(0, 50000)),
        " Total Length of Bwd Packets": str(random.uniform(0, 100000)),
        "Fwd Packet Length Max": str(random.uniform(0, 1500)),
        " Fwd Packet Length Min": str(random.uniform(0, 100)),
        " Fwd Packet Length Mean": str(random.uniform(0, 750)),
        " Fwd Packet Length Std": str(random.uniform(0, 500)),
        "Bwd Packet Length Max": str(random.uniform(0, 1500)),
        " Bwd Packet Length Min": str(random.uniform(0, 100)),
        " Bwd Packet Length Mean": str(random.uniform(0, 750)),
        " Bwd Packet Length Std": str(random.uniform(0, 500)),
        "Flow Bytes/s": str(random.uniform(0, 1000000)),
        " Flow Packets/s": str(random.uniform(0, 50000)),
        " FIN Flag Count": str(random.choice([0, 0, 0, 1])),
        " SYN Flag Count": str(random.choice([0, 1])),
        " RST Flag Count": str(random.choice([0, 0, 0, 0, 1])),
        " PSH Flag Count": str(random.choice([0, 0, 1])),
        " ACK Flag Count": str(random.choice([0, 1])),
        " URG Flag Count": str(0),
        "Flow IAT Mean": str(random.uniform(0, 5000000)),
        " Flow IAT Std": str(random.uniform(0, 3000000)),
        " Flow IAT Max": str(random.uniform(0, 10000000)),
        " Flow IAT Min": str(random.uniform(0, 100000)),
        "Fwd IAT Mean": str(random.uniform(0, 5000000)),
        " Fwd IAT Std": str(random.uniform(0, 3000000)),
        " Fwd IAT Max": str(random.uniform(0, 10000000)),
        " Fwd IAT Min": str(random.uniform(0, 100000)),
        "Bwd IAT Mean": str(random.uniform(0, 5000000)),
        " Bwd IAT Std": str(random.uniform(0, 3000000)),
        " Bwd IAT Max": str(random.uniform(0, 10000000)),
        " Bwd IAT Min": str(random.uniform(0, 100000)),
        " Down/Up Ratio": str(random.uniform(0, 5)),
        " Average Packet Size": str(random.uniform(0, 1000)),
        " Avg Fwd Segment Size": str(random.uniform(0, 750)),
        " Avg Bwd Segment Size": str(random.uniform(0, 750)),
    }


def generate_attack_row(attack_type: str) -> dict[str, str]:
    """Generate an attack flow with characteristics specific to the attack type."""
    row = generate_benign_row()

    if attack_type.startswith("DDoS") or attack_type.startswith("DoS"):
        # High packet rates, short flows
        row[" Total Fwd Packets"] = str(random.randint(100, 10000))
        row[" Flow Packets/s"] = str(random.uniform(50000, 500000))
        row["Flow Bytes/s"] = str(random.uniform(1000000, 50000000))
        row[" Flow Duration"] = str(random.randint(0, 5000000))
        row[" SYN Flag Count"] = str(random.choice([1, 1, 1, 0]))
        row[" Destination Port"] = str(80)
    elif attack_type == "PortScan":
        # Many short flows to different ports
        row[" Destination Port"] = str(random.randint(1, 65535))
        row[" Total Fwd Packets"] = str(random.randint(1, 5))
        row[" Total Backward Packets"] = str(random.randint(0, 2))
        row[" Flow Duration"] = str(random.randint(0, 1000000))
        row[" SYN Flag Count"] = str(1)
        row[" RST Flag Count"] = str(random.choice([0, 1]))
    elif "Patator" in attack_type:
        # Brute force: many connection attempts
        row[" Destination Port"] = str(21 if "FTP" in attack_type else 22)
        row[" Total Fwd Packets"] = str(random.randint(5, 50))
        row[" Flow Duration"] = str(random.randint(1000000, 30000000))
    elif attack_type == "Bot":
        # Periodic, consistent patterns
        row[" Flow Duration"] = str(random.randint(10000000, 120000000))
        row["Flow IAT Mean"] = str(random.uniform(100000, 1000000))
        row[" Flow IAT Std"] = str(random.uniform(0, 50000))
    elif attack_type.startswith("Web Attack"):
        row[" Destination Port"] = str(80)
        row[" Total Fwd Packets"] = str(random.randint(10, 200))
        row[" PSH Flag Count"] = str(1)

    return row


def generate_fixture(output_path: Path, num_rows: int = 3000) -> None:
    """Generate the fixture CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels = list(ATTACK_WEIGHTS.keys())
    weights = list(ATTACK_WEIGHTS.values())

    all_columns = ID_COLUMNS + COLUMNS

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_columns)
        writer.writeheader()

        for i in range(num_rows):
            label = random.choices(labels, weights=weights, k=1)[0]
            day_idx = min(i * len(DAYS) // num_rows, len(DAYS) - 1)
            day = DAYS[day_idx]
            hour = random.randint(8, 17)
            minute = random.randint(0, 59)
            second = random.randint(0, 59)
            timestamp = f"{day} {hour}:{minute:02d}:{second:02d}"

            src_ip = random.choice(IPS[:8])
            dst_ip = random.choice(IPS[8:])
            src_port = random.randint(1024, 65535)
            protocol = random.choice([6, 17])  # TCP or UDP

            row = generate_benign_row() if label == "BENIGN" else generate_attack_row(label)

            row[" Label"] = label
            row["Flow ID"] = f"{src_ip}-{dst_ip}-{src_port}-{row[' Destination Port']}-{protocol}"
            row[" Source IP"] = src_ip
            row[" Source Port"] = str(src_port)
            row[" Destination IP"] = dst_ip
            row[" Protocol"] = str(protocol)
            row[" Timestamp"] = timestamp

            writer.writerow(row)


if __name__ == "__main__":
    fixture_path = Path(__file__).resolve().parents[4] / "data" / "fixtures" / "cicids2017_sample.csv"
    generate_fixture(fixture_path)
    print(f"Generated fixture at {fixture_path}")
