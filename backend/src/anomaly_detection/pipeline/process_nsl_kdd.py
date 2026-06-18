"""NSL-KDD dataset download and processing pipeline.

Downloads KDDTrain+.txt and KDDTest+.txt from the canonical GitHub mirror,
encodes categorical features, computes 7 extra derived features, and
produces the same parquet artifacts expected by the rest of the pipeline.

Usage:
    python -m anomaly_detection.pipeline.process_nsl_kdd
    python -m anomaly_detection.pipeline.process_nsl_kdd --output data/processed
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from anomaly_detection.logging import get_logger, setup_logging

logger = get_logger(__name__)

# ── Dataset download URLs ──────────────────────────────────────────────────
TRAIN_URL = "https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/master/KDDTrain%2B.txt"
TEST_URL  = "https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/master/KDDTest%2B.txt"

# ── Column definitions (41 NSL-KDD features, no header in file) ───────────
NSL_COLUMNS = [
    "duration", "protocol_type", "service", "flag",
    "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent",
    "hot", "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login",
    "count", "srv_count",
    "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    "label", "difficulty",
]

# ── Fixed categorical encodings (deterministic at both train and inference) ─
PROTOCOL_MAP: dict[str, float] = {"icmp": 0.0, "tcp": 1.0, "udp": 2.0}

SERVICE_MAP: dict[str, float] = {
    "aol": 0, "auth": 1, "bgp": 2, "courier": 3, "csnet_ns": 4, "ctf": 5,
    "daytime": 6, "discard": 7, "domain": 8, "domain_u": 9, "echo": 10,
    "eco_i": 11, "ecr_i": 12, "efs": 13, "exec": 14, "finger": 15,
    "ftp": 16, "ftp_data": 17, "gopher": 18, "harvest": 19, "hostnames": 20,
    "http": 21, "http_2784": 22, "http_443": 23, "http_8001": 24, "imap4": 25,
    "IRC": 26, "iso_tsap": 27, "klogin": 28, "kshell": 29, "ldap": 30,
    "link": 31, "login": 32, "mtp": 33, "name": 34, "netbios_dgm": 35,
    "netbios_ns": 36, "netbios_ssn": 37, "netstat": 38, "nnsp": 39,
    "nntp": 40, "ntp_u": 41, "other": 42, "pm_dump": 43, "pop_2": 44,
    "pop_3": 45, "printer": 46, "private": 47, "red_i": 48, "remote_job": 49,
    "rje": 50, "shell": 51, "smtp": 52, "sql_net": 53, "ssh": 54,
    "sunrpc": 55, "supdup": 56, "systat": 57, "telnet": 58, "tftp_u": 59,
    "tim_channel": 60, "time": 61, "urh_i": 62, "urp_i": 63, "uucp": 64,
    "uucp_path": 65, "vmnet": 66, "whois": 67, "X11": 68, "Z39_50": 69,
}

FLAG_MAP: dict[str, float] = {
    "OTH": 0, "REJ": 1, "RSTO": 2, "RSTOS0": 3, "RSTR": 4,
    "S0": 5, "S1": 6, "S2": 7, "S3": 8, "SF": 9, "SH": 10,
}

# ── Attack category grouping ───────────────────────────────────────────────
DOS_ATTACKS = {
    "back", "land", "neptune", "pod", "smurf", "teardrop",
    "apache2", "udpstorm", "processtable", "worm",
}
PROBE_ATTACKS = {"satan", "ipsweep", "nmap", "portsweep", "mscan", "saint"}
R2L_ATTACKS = {
    "guess_passwd", "ftp_write", "imap", "phf", "multihop",
    "warezmaster", "warezclient", "spy", "xlock", "xsnoop",
    "snmpguess", "snmpgetattack", "httptunnel", "sendmail", "named",
}
U2R_ATTACKS = {"buffer_overflow", "loadmodule", "rootkit", "perl", "sqlattack", "xterm", "ps"}


def map_attack_label(raw: str) -> str:
    name = raw.lower().strip()
    if name == "normal":
        return "BENIGN"
    if name in DOS_ATTACKS:
        return "DoS"
    if name in PROBE_ATTACKS:
        return "Probe"
    if name in R2L_ATTACKS:
        return "R2L"
    if name in U2R_ATTACKS:
        return "U2R"
    return "Unknown"


def download_file(url: str, dest: Path) -> None:
    if dest.exists():
        logger.info("already_downloaded", path=str(dest))
        return
    logger.info("downloading", url=url, dest=str(dest))
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)
    logger.info("download_complete", path=str(dest))


def load_nsl_kdd(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, names=NSL_COLUMNS)
    logger.info("loaded_nsl_kdd", path=str(path), rows=len(df))
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["protocol_type"] = df["protocol_type"].map(PROTOCOL_MAP).fillna(0.0)
    df["service"] = df["service"].map(SERVICE_MAP).fillna(42.0)   # 42 = "other"
    df["flag"] = df["flag"].map(FLAG_MAP).fillna(9.0)              # 9 = "SF"
    return df


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute 7 extra features that complement the 41 standard NSL-KDD features."""
    df = df.copy()

    dur = df["duration"].clip(lower=0) + 1e-6

    # 1. packet_rate: estimated packets per second (count / duration)
    df["packet_rate"] = df["count"] / dur

    # 2. byte_rate: total bytes transferred per second
    df["byte_rate"] = (df["src_bytes"] + df["dst_bytes"]) / dur

    # 3. avg_packet_size: average bytes per connection in the window
    df["avg_packet_size"] = (df["src_bytes"] + df["dst_bytes"]) / (df["count"].clip(lower=1))

    # 4. flow_duration: same as duration but kept explicitly for model clarity
    df["flow_duration"] = df["duration"].clip(lower=0)

    # 5. inter_arrival_time: average time between successive packets
    df["inter_arrival_time"] = df["duration"] / (df["count"].clip(lower=1))

    # 6. fwd_bwd_ratio: forward-to-backward byte ratio (asymmetry indicator)
    df["fwd_bwd_ratio"] = df["src_bytes"] / (df["dst_bytes"] + 1.0)

    # 7. port_entropy: proxy for connection diversity (srv_diff_host_rate * diff_srv_rate)
    df["port_entropy"] = df["srv_diff_host_rate"] * df["diff_srv_rate"] * 10.0

    return df


def process_dataset(raw_dir: Path, output_dir: Path) -> None:
    """Full NSL-KDD processing pipeline.

    Downloads data, encodes features, computes extras, scales, and
    writes train_benign.parquet, train.parquet, test.parquet, scaler.joblib.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download
    train_path = raw_dir / "KDDTrain+.txt"
    test_path  = raw_dir / "KDDTest+.txt"
    download_file(TRAIN_URL, train_path)
    download_file(TEST_URL,  test_path)

    # Load
    train_raw = load_nsl_kdd(train_path)
    test_raw  = load_nsl_kdd(test_path)

    # Map labels
    train_raw["label"] = train_raw["label"].apply(map_attack_label)
    test_raw["label"]  = test_raw["label"].apply(map_attack_label)

    # Encode categoricals
    train_enc = encode_categoricals(train_raw)
    test_enc  = encode_categoricals(test_raw)

    # Derived features
    train_enc = add_derived_features(train_enc)
    test_enc  = add_derived_features(test_enc)

    # Import here to avoid circular import at module load time
    from anomaly_detection.constants import FEATURE_COLUMNS

    # Validate all features exist
    missing = [f for f in FEATURE_COLUMNS if f not in train_enc.columns]
    if missing:
        raise ValueError(f"Missing features after processing: {missing}")

    # Keep only features + label
    train_df = train_enc[FEATURE_COLUMNS + ["label"]].copy()
    test_df  = test_enc[FEATURE_COLUMNS + ["label"]].copy()

    # Drop rows with NaN / Inf
    import numpy as np
    for df in [train_df, test_df]:
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
    train_df.dropna(inplace=True)
    test_df.dropna(inplace=True)

    logger.info("processed_split", train=len(train_df), test=len(test_df))

    # Fit scaler on TRAINING data only
    scaler = StandardScaler()
    X_train = train_df[FEATURE_COLUMNS].values
    scaler.fit(X_train)

    # Benign-only subset for unsupervised models
    train_benign = train_df[train_df["label"] == "BENIGN"].copy()

    # Save artefacts
    train_benign.to_parquet(output_dir / "train_benign.parquet", index=False)
    train_df.to_parquet(output_dir / "train.parquet", index=False)
    test_df.to_parquet(output_dir / "test.parquet", index=False)
    joblib.dump(scaler, output_dir / "scaler.joblib")

    logger.info(
        "pipeline_complete",
        train_benign=len(train_benign),
        train_full=len(train_df),
        test=len(test_df),
        features=len(FEATURE_COLUMNS),
        output_dir=str(output_dir),
    )

    # Label distribution
    logger.info("train_label_dist", **train_df["label"].value_counts().to_dict())
    logger.info("test_label_dist",  **test_df["label"].value_counts().to_dict())


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and process NSL-KDD dataset")
    parser.add_argument("--raw-dir",  type=Path, default=Path("data/raw/nsl_kdd"))
    parser.add_argument("--output",   type=Path, default=Path("data/processed"))
    args = parser.parse_args()

    setup_logging("INFO")
    process_dataset(args.raw_dir, args.output)


if __name__ == "__main__":
    main()
