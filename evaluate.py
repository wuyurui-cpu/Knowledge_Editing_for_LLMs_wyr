import argparse
import csv
from pathlib import Path
from typing import Dict, Any, Optional, List


def read_json(filepath: str | Path) -> Any:
    with Path(filepath).open("r", encoding="utf-8") as f:
        import json
        return json.load(f)


def safe_mean(values) -> Optional[float]:
    """计算均值，若可迭代为空则返回 None"""
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def extract_numbers(obj: Any) -> List[float]:
    """递归提取可迭代对象中的数值"""
    if isinstance(obj, (int, float)):
        return [float(obj)]
    if isinstance(obj, (list, tuple)):
        result = []
        for item in obj:
            result.extend(extract_numbers(item))
        return result
    if isinstance(obj, dict):
        result = []
        for v in obj.values():
            result.extend(extract_numbers(v))
        return result
    return []


def gather_metric_entries(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """根据算法类型提取 metrics 条目"""
    algo = data.get("algorithm")
    if algo == "ROME":
        entries = []
        for case in data.get("case_results", []):
            metric = case.get("metrics", [])
            if isinstance(metric, list):
                entries.extend(metric)
            elif isinstance(metric, dict):
                entries.append(metric)
        return entries

    metric_val = data.get("metrics", [])
    if isinstance(metric_val, list):
        return metric_val
    if isinstance(metric_val, dict):
        return [metric_val]
    return []


def compute_summary(input_path: str) -> Dict[str, Any]:
    raw = read_json(input_path)
    metric_items = gather_metric_entries(raw)

    # 提取所有的 post 子字典
    post_dicts = [m.get("post", {}) for m in metric_items if isinstance(m, dict)]

    # 收集各种数值
    rewrite_scores = []
    rephrase_scores = []
    locality_scores = []

    for pdict in post_dicts:
        rewrite_scores.extend(extract_numbers(pdict.get("rewrite_acc")))
        rephrase_scores.extend(extract_numbers(pdict.get("rephrase_acc")))
        locality_scores.extend(extract_numbers(pdict.get("locality", {})))

    return {
        "algorithm": raw.get("algorithm", Path(input_path).stem),
        "count": raw.get("num_records"),
        "edit_score": safe_mean(rewrite_scores),
        "rephrase_score": safe_mean(rephrase_scores),
        "neighborhood_score": safe_mean(locality_scores),
        "duration_sec": raw.get("elapsed_seconds"),
        "peak_mem_gb": raw.get("peak_memory_gb"),
        "origin_path": input_path,
    }



def as_percentage(value: Optional[float]) -> str:
    return "" if value is None else f"{value * 100:.2f}%"



def main() -> None:
    cli = argparse.ArgumentParser(description="Aggregate and display ES/PS/NS metrics.")
    cli.add_argument("--files", nargs="+", default=["output/rome_output.json", "output/memit_output.json"])
    cli.add_argument("--output", default="results/final_metrics.csv")
    args = cli.parse_args()

    # 只处理存在的文件
    summaries = [compute_summary(f) for f in args.files if Path(f).exists()]

    # 确保输出目录存在
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # 写入 CSV（表头使用不同的名称）
    with Path(args.output).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "Algorithm",
                "NumRecords",
                "EditSuccess",
                "RephraseSuccess",
                "NeighborhoodSuccess",
                "Time(s)",
                "PeakMem(GB)",
                "SourceFile",
            ],
        )
        writer.writeheader()
        for row in summaries:
            writer.writerow({
                "Algorithm": row["algorithm"],
                "NumRecords": row["count"],
                "EditSuccess": row["edit_score"],
                "RephraseSuccess": row["rephrase_score"],
                "NeighborhoodSuccess": row["neighborhood_score"],
                "Time(s)": row["duration_sec"],
                "PeakMem(GB)": row["peak_mem_gb"],
                "SourceFile": row["origin_path"],
            })

    # 打印简化的纯文本表格（不采用 Markdown，改用对齐文本）
    print("\n" + "=" * 80)
    print(f"{'Algorithm':<12} {'Records':>8} {'ES':>10} {'PS':>10} {'NS':>10} {'Time(s)':>10} {'PeakMem':>10}")
    print("-" * 80)
    for rec in summaries:
        es_str = as_percentage(rec["edit_score"])
        ps_str = as_percentage(rec["rephrase_score"])
        ns_str = as_percentage(rec["neighborhood_score"])
        time_str = f"{rec['duration_sec']:.2f}" if rec["duration_sec"] is not None else ""
        mem_str = f"{rec['peak_mem_gb']:.2f}" if rec["peak_mem_gb"] is not None else ""
        print(f"{rec['algorithm']:<12} {rec['count']:>8} {es_str:>10} {ps_str:>10} {ns_str:>10} {time_str:>10} {mem_str:>10}")
    print("=" * 80)


if __name__ == "__main__":
    main()