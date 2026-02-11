#!/usr/bin/env python3
"""
AWS EC2/Batch usage and pricing report.

Produces a CSV with instance type, hours, cost, $/instance-hour, vCPUs, and $/vCPU-hour,
plus total cost. Uses Cost Explorer (cost/usage) and EC2 (vCPU per instance type).

Note: AWS billing data is typically delayed by up to 24 hours. Costs and usage for
today (and sometimes yesterday) may be incomplete. For the most accurate picture,
use a date range that ends at least one full day in the past.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import boto3
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import seaborn as sns
from botocore.exceptions import ClientError
from platformdirs import user_data_dir


# Cost Explorer is a global API; convention is us-east-1
CE_REGION = "us-east-1"

# default output location (same tree as benchmark results)
RESULTS_DIR = Path(user_data_dir("glow", "glow_author")) / "results"

# EC2 service names in Cost Explorer (on-demand vs spot)
EC2_SERVICES = [
    "Amazon Elastic Compute Cloud - Compute",
    "Amazon EC2",
]


def get_ec2_cost_and_usage_daily(
    ce_client,
    start_date: str,
    end_date: str,
) -> list[dict]:
    """Fetch daily cost and usage from Cost Explorer grouped by instance type.

    Returns a list of dicts, one per (date, instance_type) pair:
        {"Date": "YYYY-MM-DD", "InstanceType": str, "Cost": float, "UsageQuantity": float}
    """
    filter_expr = {
        "Or": [
            {"Dimensions": {"Key": "SERVICE", "Values": [svc]}} for svc in EC2_SERVICES
        ]
    }
    records: list[dict] = []
    next_token = None

    while True:
        params = {
            "TimePeriod": {"Start": start_date, "End": end_date},
            "Granularity": "DAILY",
            "Metrics": ["UnblendedCost", "UsageQuantity"],
            "Filter": filter_expr,
            "GroupBy": [{"Type": "DIMENSION", "Key": "INSTANCE_TYPE"}],
        }
        if next_token:
            params["NextPageToken"] = next_token

        try:
            response = ce_client.get_cost_and_usage(**params)
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            if code == "SubscriptionRequiredException":
                print(
                    "Error: Cost Explorer is not enabled for this account. "
                    "Enable it in Billing preferences in the AWS console.",
                    file=sys.stderr,
                )
            elif code in ("AccessDeniedException", "UnauthorizedException"):
                print(
                    "Error: This identity does not have permission to use Cost Explorer. "
                    "Requires ce:GetCostAndUsage (e.g. AWSBillingReadOnlyAccess or "
                    "a custom policy).",
                    file=sys.stderr,
                )
            else:
                print(f"Cost Explorer error: {e}", file=sys.stderr)
            raise SystemExit(1) from e

        for result in response.get("ResultsByTime", []):
            date = result["TimePeriod"]["Start"]
            for group in result.get("Groups", []):
                keys = group.get("Keys", [])
                metrics = group.get("Metrics", {})
                if not keys:
                    continue
                instance_type = keys[0]
                # skip non-instance entries (EBS, data transfer, etc.)
                if not instance_type or "." not in instance_type:
                    continue
                cost = float(metrics.get("UnblendedCost", {}).get("Amount", 0))
                usage = float(metrics.get("UsageQuantity", {}).get("Amount", 0))
                records.append(
                    {
                        "Date": date,
                        "InstanceType": instance_type,
                        "Cost": cost,
                        "UsageQuantity": usage,
                    }
                )

        next_token = response.get("NextPageToken")
        if not next_token:
            break

    return records


def aggregate_by_instance_type(records: list[dict]) -> list[dict]:
    """Collapse daily records into one row per instance type."""
    agg: dict[str, dict] = {}
    for r in records:
        it = r["InstanceType"]
        if it not in agg:
            agg[it] = {"InstanceType": it, "Cost": 0.0, "UsageQuantity": 0.0}
        agg[it]["Cost"] += r["Cost"]
        agg[it]["UsageQuantity"] += r["UsageQuantity"]
    return list(agg.values())


def get_vcpus_for_instance_types(
    ec2_client, instance_types: list[str]
) -> dict[str, int]:
    """Return mapping of instance_type -> vCPU count. Unknown types get 0."""
    result = {}
    # describe_instance_types accepts up to 100 at a time
    for i in range(0, len(instance_types), 100):
        chunk = instance_types[i : i + 100]
        try:
            resp = ec2_client.describe_instance_types(InstanceTypes=chunk)
            for it in resp.get("InstanceTypes", []):
                name = it.get("InstanceType")
                vcpu_info = it.get("VCpuInfo", {})
                result[name] = int(vcpu_info.get("DefaultVCpus", 0))
        except ClientError:
            # batch call failed (e.g. one invalid type) — fall back to one-by-one
            for itype in chunk:
                if itype in result:
                    continue
                try:
                    resp = ec2_client.describe_instance_types(InstanceTypes=[itype])
                    for it in resp.get("InstanceTypes", []):
                        vcpu_info = it.get("VCpuInfo", {})
                        result[itype] = int(vcpu_info.get("DefaultVCpus", 0))
                except ClientError:
                    pass
        for it in chunk:
            if it not in result:
                result[it] = 0
    return result


def plot_daily_cost(records: list[dict], vcpus_map: dict[str, int],
                    out_dir: Path) -> Path:
    """Generate daily + cumulative cost subplots. Returns saved path."""
    sns.set_theme(style="darkgrid")

    # aggregate by date
    daily_cost: dict[str, float] = defaultdict(float)
    total_vcpu_hours = 0.0
    total_cost = 0.0
    for r in records:
        daily_cost[r["Date"]] += r["Cost"]
        total_cost += r["Cost"]
        vcpus = vcpus_map.get(r["InstanceType"], 0)
        total_vcpu_hours += r["UsageQuantity"] * vcpus

    avg_per_vcpu_hour = total_cost / total_vcpu_hours if total_vcpu_hours else 0

    dates_str = sorted(daily_cost.keys())
    dates = [datetime.strptime(d, "%Y-%m-%d").date() for d in dates_str]
    costs = [daily_cost[d] for d in dates_str]
    cum_cost = np.cumsum(costs)

    out_dir.mkdir(parents=True, exist_ok=True)

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # ── top: daily cost bars ────────────────────────────────────────────
    ax_top.bar(dates, costs, color="#4c9ed9", alpha=0.85)
    ax_top.set_ylabel("Daily cost ($)")

    # ── bottom: cumulative cost line ────────────────────────────────────
    ax_bot.plot(dates, cum_cost, color="#e05050", linewidth=2.2)
    ax_bot.fill_between(dates, cum_cost, alpha=0.15, color="#e05050")
    ax_bot.set_ylabel("Cumulative cost ($)")
    ax_bot.set_xlabel("Date")

    # date formatting (shared x-axis, only label bottom)
    ax_bot.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax_bot.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate(rotation=45)

    # title with average $/vCPU-hour
    subtitle = (f"total: ${total_cost:.2f}   |   "
                f"avg cost per vCPU-hour: ${avg_per_vcpu_hour:.4f}")
    fig.suptitle("EC2 Daily Spend", fontsize=13, fontweight="bold")
    ax_top.set_title(subtitle, fontsize=9, color="0.35", pad=4)

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    p = out_dir / "usage_report.pdf"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export AWS EC2/Batch usage and pricing to CSV.",
        epilog=(
            "Billing data is typically delayed by up to 24 hours. "
            "For accurate totals, use an end date at least one day in the past."
        ),
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Start date (YYYY-MM-DD). Default: 30 days before --end.",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="End date (YYYY-MM-DD, exclusive). Default: today.",
    )
    parser.add_argument(
        "--region",
        type=str,
        default="us-east-1",
        help="Region for EC2 instance type lookup (default: us-east-1).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output CSV path. Default: <results_dir>/usage_report.csv.",
    )
    args = parser.parse_args()

    today = datetime.now(timezone.utc).date()
    end = args.end
    if end is None:
        end = today.isoformat()
    else:
        try:
            datetime.strptime(end, "%Y-%m-%d")
        except ValueError:
            print(f"Invalid --end date: {end}. Use YYYY-MM-DD.", file=sys.stderr)
            raise SystemExit(1) from None

    start = args.start
    if start is None:
        end_dt = datetime.strptime(end, "%Y-%m-%d").date()
        start_dt = end_dt - timedelta(days=30)
        start = start_dt.isoformat()
    else:
        try:
            datetime.strptime(start, "%Y-%m-%d")
        except ValueError:
            print(f"Invalid --start date: {start}. Use YYYY-MM-DD.", file=sys.stderr)
            raise SystemExit(1) from None

    # Billing delay notice
    end_dt = datetime.strptime(end, "%Y-%m-%d").date()
    if end_dt >= today or (today - end_dt).days < 2:
        print(
            "Note: AWS billing data is typically delayed by up to 24 hours.\n"
            "      Costs for recent days may be incomplete.\n",
        )

    ce = boto3.client("ce", region_name=CE_REGION)
    ec2 = boto3.client("ec2", region_name=args.region)

    print(f"Fetching EC2 cost & usage from {start} to {end} ...")
    daily_records = get_ec2_cost_and_usage_daily(ce, start, end)
    if not daily_records:
        print("No EC2 instance usage found for this period.")
        return

    groups = aggregate_by_instance_type(daily_records)
    instance_types = list({g["InstanceType"] for g in groups})
    vcpus_map = get_vcpus_for_instance_types(ec2, instance_types)

    # Build rows
    rows: list[dict] = []
    total_cost = 0.0
    total_hours = 0.0
    for g in groups:
        it = g["InstanceType"]
        hours = g["UsageQuantity"]
        cost = g["Cost"]
        total_cost += cost
        total_hours += hours
        usd_per_instance_hour = cost / hours if hours else 0
        vcpus = vcpus_map.get(it, 0)
        vcpu_hours = hours * vcpus if vcpus else 0
        usd_per_vcpu_hour = cost / vcpu_hours if vcpu_hours else 0
        rows.append(
            {
                "instance_type": it,
                "hours": hours,
                "cost_usd": cost,
                "usd_per_instance_hour": usd_per_instance_hour,
                "vcpus": vcpus,
                "usd_per_vcpu_hour": usd_per_vcpu_hour,
            }
        )

    # Sort by cost descending
    rows.sort(key=lambda r: r["cost_usd"], reverse=True)

    # ── pretty-print table to terminal ──────────────────────────────────
    headers = ["Instance Type", "Hours", "Cost ($)", "$/Inst-Hr", "vCPUs", "$/vCPU-Hr"]
    fmt_rows = []
    for r in rows:
        fmt_rows.append([
            r["instance_type"],
            f'{r["hours"]:.2f}',
            f'{r["cost_usd"]:.4f}',
            f'{r["usd_per_instance_hour"]:.4f}',
            str(r["vcpus"]) if r["vcpus"] else "",
            f'{r["usd_per_vcpu_hour"]:.4f}' if r["vcpus"] else "",
        ])

    # column widths
    col_w = [len(h) for h in headers]
    for row in fmt_rows:
        for i, cell in enumerate(row):
            col_w[i] = max(col_w[i], len(cell))

    def fmt_line(cells):
        parts = []
        for i, cell in enumerate(cells):
            # left-align instance type, right-align numbers
            if i == 0:
                parts.append(cell.ljust(col_w[i]))
            else:
                parts.append(cell.rjust(col_w[i]))
        return "  ".join(parts)

    sep = "  ".join("-" * w for w in col_w)
    print()
    print(fmt_line(headers))
    print(sep)
    for row in fmt_rows:
        print(fmt_line(row))
    print(sep)
    print(fmt_line(["TOTAL", f"{total_hours:.2f}", f"{total_cost:.2f}", "", "", ""]))
    print()

    # ── write CSV ───────────────────────────────────────────────────────
    csv_path = args.output
    if csv_path is None:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        csv_path = RESULTS_DIR / "usage_report.csv"

    fieldnames = [
        "instance_type",
        "hours",
        "cost_usd",
        "usd_per_instance_hour",
        "vcpus",
        "usd_per_vcpu_hour",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "instance_type": r["instance_type"],
                    "hours": f'{r["hours"]:.4f}',
                    "cost_usd": f'{r["cost_usd"]:.4f}',
                    "usd_per_instance_hour": f'{r["usd_per_instance_hour"]:.6f}',
                    "vcpus": r["vcpus"],
                    "usd_per_vcpu_hour": (
                        f'{r["usd_per_vcpu_hour"]:.6f}' if r["vcpus"] else ""
                    ),
                }
            )
        writer.writerow({})
        writer.writerow(
            {
                "instance_type": "TOTAL",
                "hours": f"{total_hours:.2f}",
                "cost_usd": f"{total_cost:.2f}",
            }
        )

    print(f"CSV saved: {csv_path}")

    # ── plot ────────────────────────────────────────────────────────────
    plot_path = plot_daily_cost(daily_records, vcpus_map, csv_path.parent)
    print(f"Plot saved: {plot_path}")


if __name__ == "__main__":
    main()
