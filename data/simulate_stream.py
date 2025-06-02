"""
simulate_stream.py — Build 10 time-ordered query batches from ML Q&A data.

Batch structure:
  1-5:  clean ML Q&A (in-distribution)
  6-8:  70% out-of-domain + 30% ML Q&A  <- drift window
  9-10: clean ML Q&A (recovered)

Output: data/batches/batch_01.jsonl ... batch_10.jsonl

Usage:
    python data/simulate_stream.py
    python data/simulate_stream.py --batch-size 50 --seed 42
"""

import argparse
import json
import random
from pathlib import Path

BATCH_SIZE = 50
OUT_DIR = Path(__file__).parent / "batches"

# Out-of-domain queries (manufacturing, finance, healthcare, generic)
# These share vocabulary with ML Q&A (Python, data, model) but are different domain
OOD_QUERIES = [
    # Manufacturing / IoT
    "How do I read sensor data from a Siemens PLC using Python?",
    "What is the difference between SCADA and DCS in industrial control systems?",
    "How do I implement a PID controller for temperature regulation?",
    "What protocols does OPC-UA support for industrial communication?",
    "How can I detect anomalies in vibration sensor readings from rotating machinery?",
    "What is the best way to store time-series data from 500 IoT sensors?",
    "How do I connect a Modbus RTU device to my Python application?",
    "What is the difference between preventive and predictive maintenance?",
    "How do I implement real-time quality control on a production line?",
    "What are the key metrics to monitor on a CNC milling machine?",
    "How do I handle network latency in a factory automation system?",
    "What is the standard for PLC programming (IEC 61131-3)?",
    "How do I synchronize data from multiple sensors with different sampling rates?",
    "What is OEE (Overall Equipment Effectiveness) and how is it calculated?",
    "How do I implement alarm management in a SCADA system?",
    "What is the best database for storing high-frequency sensor data (10kHz)?",
    "How do I detect tool wear from spindle current measurements?",
    "What is digital twin and how is it used in manufacturing?",
    "How do I implement predictive maintenance using vibration analysis?",
    "What is the difference between edge computing and cloud computing for IoT?",
    # Finance / trading
    "How do I calculate the Sharpe ratio for a portfolio of assets?",
    "What is the difference between VaR and CVaR in risk management?",
    "How do I implement a mean-reversion trading strategy in Python?",
    "What is the Black-Scholes model and when does it break down?",
    "How do I backtest a momentum trading strategy without lookahead bias?",
    "What are the main risk factors in the Fama-French 3-factor model?",
    "How do I calculate implied volatility from options prices?",
    "What is the Kelly criterion and how do I use it for position sizing?",
    "How do I detect regime changes in financial time series?",
    "What is the difference between cointegration and correlation?",
    "How do I implement pairs trading with error correction models?",
    "What is GARCH and when should I use it for volatility modeling?",
    "How do I calculate beta for a stock relative to the S&P 500?",
    "What is the difference between P&L attribution and performance attribution?",
    "How do I implement a limit order book simulator?",
    "What is market microstructure and why does it matter for algorithmic trading?",
    "How do I handle survivorship bias in backtesting?",
    "What are the key considerations for latency in high-frequency trading?",
    "How do I model credit default risk for a bond portfolio?",
    "What is the difference between futures and forwards contracts?",
    # Healthcare / clinical
    "How do I calculate sensitivity and specificity for a diagnostic test?",
    "What is the difference between ITT and per-protocol analysis in clinical trials?",
    "How do I handle missing data in longitudinal clinical trial data?",
    "What is CONSORT and why do clinical trials need to follow it?",
    "How do I calculate sample size for a randomized controlled trial?",
    "What is the difference between HIPAA and GDPR for healthcare data?",
    "How do I implement ICD-10 code mapping in an EHR system?",
    "What is a Kaplan-Meier survival curve and how is it interpreted?",
    "How do I detect drug interactions in prescription data?",
    "What is the NNT (number needed to treat) and how is it calculated?",
    "How do I de-identify patient records while preserving utility for research?",
    "What is the difference between Type I and Type II errors in medical testing?",
    "How do I implement HL7 FHIR integration for medical records?",
    "What are the main regulatory hurdles for AI diagnostic tools (FDA)?",
    "How do I calculate concordance index (C-statistic) for survival models?",
    "What is propensity score matching and when should I use it?",
    "How do I handle imbalanced classes in a rare disease classifier?",
    "What is the difference between real-world evidence and RCT data?",
    "How do I implement audit trails for clinical decision support software?",
    "What is the difference between screening and diagnostic sensitivity?",
    # Supply chain / logistics
    "How do I implement demand forecasting for seasonal inventory?",
    "What is the economic order quantity (EOQ) formula?",
    "How do I optimize warehouse slotting for pick frequency?",
    "What is the difference between MRP and MRP II in supply chain planning?",
    "How do I calculate safety stock for uncertain lead times?",
    "What is a bullwhip effect and how do I reduce it in a supply chain?",
    "How do I implement vehicle routing optimization for last-mile delivery?",
    "What is the difference between 3PL and 4PL logistics providers?",
    "How do I measure fill rate and service level in inventory management?",
    "What is collaborative planning, forecasting and replenishment (CPFR)?",
    "How do I model supply chain disruption risk?",
    "What is the difference between push and pull supply chain strategies?",
    "How do I implement ABC-XYZ analysis for inventory classification?",
    "What are the key performance indicators for a distribution center?",
    "How do I calculate days of inventory outstanding (DIO)?",
    # Generic / consumer tech
    "How do I implement infinite scroll in a React application?",
    "What is the difference between REST and GraphQL APIs?",
    "How do I implement OAuth 2.0 authorization code flow?",
    "What is the best way to handle database migrations in production?",
    "How do I implement real-time notifications with WebSockets?",
    "What is the difference between Docker and virtual machines?",
    "How do I implement A/B testing for a mobile app feature?",
    "What is the best caching strategy for a read-heavy web application?",
    "How do I implement full-text search in PostgreSQL?",
    "What is the difference between blue-green and canary deployments?",
    "How do I implement rate limiting in an API gateway?",
    "What is eventual consistency and when is it acceptable?",
    "How do I optimize database queries that are running slowly?",
    "What is the difference between synchronous and asynchronous messaging?",
    "How do I implement idempotency for financial transactions?",
]


def load_ml_qa_queries(data_path: Path) -> list[str]:
    queries = []
    with open(data_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            q = record.get("instruction") or record.get("input") or record.get("query", "")
            if q and len(q.split()) >= 5:
                queries.append(q)
    return queries


def build_batches(
    ml_queries: list[str],
    ood_queries: list[str],
    batch_size: int,
    n_batches: int,
    seed: int,
) -> list[list[dict]]:
    rng = random.Random(seed)
    batches = []

    for batch_num in range(1, n_batches + 1):
        if batch_num <= 5 or batch_num >= 9:
            # In-distribution: all ML Q&A
            sample = rng.sample(ml_queries, min(batch_size, len(ml_queries)))
            records = [
                {"id": f"batch_{batch_num:02d}_q{i:03d}", "query": q, "domain": "ml_qa", "batch": batch_num}
                for i, q in enumerate(sample)
            ]
        else:
            # Drift window (batches 6-8): 70% OOD + 30% ML
            n_ood = int(batch_size * 0.70)
            n_ml = batch_size - n_ood
            ood_sample = rng.sample(ood_queries, min(n_ood, len(ood_queries)))
            ml_sample = rng.sample(ml_queries, min(n_ml, len(ml_queries)))
            records = []
            for i, q in enumerate(ood_sample):
                domain = _classify_ood(q)
                records.append({"id": f"batch_{batch_num:02d}_q{i:03d}", "query": q, "domain": domain, "batch": batch_num})
            for i, q in enumerate(ml_sample):
                records.append({"id": f"batch_{batch_num:02d}_q{n_ood+i:03d}", "query": q, "domain": "ml_qa", "batch": batch_num})
            rng.shuffle(records)

        batches.append(records)
    return batches


def _classify_ood(query: str) -> str:
    q = query.lower()
    if any(w in q for w in ["plc", "scada", "sensor", "modbus", "iot", "oee", "cnc", "milling", "maintenance", "vibration"]):
        return "manufacturing"
    if any(w in q for w in ["sharpe", "var", "portfolio", "trading", "volatility", "futures", "hedge", "garch", "kelly"]):
        return "finance"
    if any(w in q for w in ["clinical", "trial", "hipaa", "ehr", "fhir", "patient", "diagnostic", "survival", "nnt"]):
        return "healthcare"
    if any(w in q for w in ["inventory", "warehouse", "logistics", "supply chain", "eoq", "mrp", "3pl", "delivery"]):
        return "supply_chain"
    return "generic"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--n-batches", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_root = Path(__file__).parent.parent
    val_path = data_root.parent / "lora-finetune/data/val.jsonl"

    if not val_path.exists():
        print(f"Warning: {val_path} not found. Using OOD queries only for in-distribution batches.")
        ml_queries = [
            "What is gradient descent and how does it work?",
            "How do I implement k-means clustering in Python?",
            "What is the difference between L1 and L2 regularization?",
            "How do I choose the right learning rate for training a neural network?",
            "What is cross-entropy loss and when should I use it?",
            "How do I implement early stopping in PyTorch?",
            "What is the vanishing gradient problem?",
            "How do I debug a model that is not converging?",
            "What is batch normalization and why does it help training?",
            "How do I implement a custom loss function in TensorFlow?",
        ] * 20
    else:
        ml_queries = load_ml_qa_queries(val_path)
        print(f"Loaded {len(ml_queries)} ML Q&A queries from {val_path}")

    batches = build_batches(ml_queries, OOD_QUERIES, args.batch_size, args.n_batches, args.seed)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for i, records in enumerate(batches, start=1):
        out_path = OUT_DIR / f"batch_{i:02d}.jsonl"
        with open(out_path, "w") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")

        domains = {}
        for r in records:
            domains[r["domain"]] = domains.get(r["domain"], 0) + 1
        domain_str = ", ".join(f"{k}:{v}" for k, v in domains.items())
        print(f"  batch_{i:02d}.jsonl — {len(records)} queries [{domain_str}]")

    print(f"\nBatches written to {OUT_DIR}/")
    print("Batches 6-8 are the drift window (70% OOD). All others are in-distribution ML Q&A.")


if __name__ == "__main__":
    main()
