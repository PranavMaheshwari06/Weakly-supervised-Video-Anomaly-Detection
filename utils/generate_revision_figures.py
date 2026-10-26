"""Generate the revision figures from the canonical package and dense per-video records.

Writes only to a new directory; the Phase-6 figures are left untouched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from generate_phase6_figures import BLUE, GRAY, ORANGE, PACKAGE, configure

OUTPUT = Path("artifacts/thesis_figures/revision_r20")
CASE_RUN = "dispatch_repro_s1337"
VECTOR_LABELS = {"0,0,0": "None", "1,0,0": "P", "1,1,0": "P+M", "1,0,1": "P+F",
                 "0,1,1": "M+F", "1,1,1": "P+M+F"}


def save(fig: plt.Figure, name: str, preview: Path | None) -> None:
    fig.savefig(OUTPUT / name, format="pdf", bbox_inches="tight",
                metadata={"Title": name.removesuffix(".pdf"), "Creator": "Revision figure generator"})
    if preview:
        preview.mkdir(parents=True, exist_ok=True)
        fig.savefig(preview / name.replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)


def load_case_records(package):
    records = []
    for entry in package["phase5_temporal_system"]["record_files"]:
        path = Path(entry["path"])
        if f"/{CASE_RUN}/" not in entry["path"]:
            continue
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"hash mismatch: {path}")
        records.append(json.loads(path.read_text(encoding="utf-8")))
    return records


def select_cases(records):
    """Selection rules stated in the thesis caption; ties break on video id."""
    iou = lambda r: r["temporal_metrics"]["primary"]["iou"]
    score = lambda r: r["routing"]["binary_score"]
    eligible = [r for r in records if r["annotation_status"] == "anomaly"
                and r["temporal_metrics"]["primary"]["ap"] is not None]
    exact = [r for r in eligible if r["routing"]["binary_pred"] == 1
             and r["routing"]["gated_primary"] == r["routing"]["dispatch_target"]]
    ranked = sorted(eligible, key=lambda r: (iou(r), r["video_id"]))
    zero = [r for r in eligible if r["temporal_metrics"]["primary"]["zero_iou"]]
    normal = [r for r in records if r["annotation_status"] == "normal"]
    return [("Best case", max(exact, key=lambda r: (iou(r), r["video_id"]))),
            ("Median case", ranked[len(ranked) // 2]),
            ("Confident misplaced detection", max(zero, key=lambda r: (score(r), r["video_id"]))),
            ("False alarm on a normal video", max(normal, key=lambda r: (score(r), r["video_id"])))]


def qualitative_cases(package, preview):
    cases = select_cases(load_case_records(package))
    fig, axes = plt.subplots(4, 1, figsize=(6.2, 7.4))
    for ax, (letter, (label, record)) in zip(axes, zip("abcd", cases)):
        meta = record["metadata"]; fps, n = meta["src_fps"], meta["keep_every_n"]
        starts = np.array(record["clip_starts"]); attention = np.array(record["attention"])
        centers = (starts + meta["clip_len"] / 2) * n / fps
        for start, end in record["intervals"]["ground_truth"]:
            ax.axvspan(start / fps, end / fps, color="#D9D9D9", zorder=0)
        ax.plot(centers, attention / attention.max(), color=BLUE, linewidth=.9, zorder=2)
        for start, end in record["intervals"]["primary_ungated"]:
            ax.plot([start / fps, end / fps], [1.12, 1.12], color=ORANGE, linewidth=4,
                    solid_capstyle="butt", zorder=3)
        routing = record["routing"]; metrics = record["temporal_metrics"]["primary"]
        vector = lambda v: "".join(str(x) for x in v)
        detail = (f"P(anomaly) {routing['binary_score']:.3f}   target P/M/F {vector(routing['dispatch_target'])}"
                  f"   predicted {vector(routing['gated_primary'])}")
        if metrics["iou"] is not None:
            detail += f"   IoU {metrics['iou']:.2f}   AP {metrics['ap']:.2f}"
        name = record["video_id"].split("/")[-1].removesuffix("_x264.mp4")
        ax.set_title(f"({letter}) {label}: {name}\n{detail}", fontsize=8.5, loc="left")
        ax.set_xlim(0, meta["src_frame_count"] / fps); ax.set_ylim(0, 1.22)
        ax.set_yticks([0, .5, 1]); ax.set_ylabel("Attention", fontsize=8.5)
    axes[-1].set_xlabel("Video time (s)")
    handles = [plt.Rectangle((0, 0), 1, 1, color="#D9D9D9"),
               plt.Line2D([], [], color=BLUE, linewidth=.9),
               plt.Line2D([], [], color=ORANGE, linewidth=4)]
    fig.legend(handles, ["Annotated interval", "Attention (scaled to max 1)", "Predicted interval (rule 0.5)"],
               loc="lower center", ncol=3, frameon=False, fontsize=8.5, bbox_to_anchor=(.5, -.01))
    fig.tight_layout(rect=(0, .03, 1, 1)); save(fig, "fig_qualitative_cases.pdf", preview)
    return [record["video_id"] for _, record in cases]


def routing_comparison(package, preview):
    fixed = package["phase1_fixed_bag_dispatch"]["derived_matched_conditions"]
    dense = package["phase5_temporal_system"]["three_checkpoint_statistics"]
    base = dense["routing_baselines"]
    panels = [
        ("All test videos (n=290)",
         ["Learned\nraw", "Learned\nbinary-gated", "All\nservices", "Binary-gated\nall services"],
         [fixed["raw_all_290"], fixed["binary_gated_all_290"], fixed["all_services_all_290"],
          fixed["binary_gated_all_services_all_290"]],
         [dense["routing"]["primary"]["raw"], dense["routing"]["primary"]["gated"],
          base["all"]["all_services"], base["all"]["binary_gated_all_services"]]),
        ("Anomalous videos only (n=140)",
         ["Learned\nraw", "All\nservices", "Binary-gated\nall services"],
         [fixed["conditional_anomaly_140"], fixed["all_services_conditional_140"],
          fixed["binary_gated_all_services_conditional_140"]],
         [dense["routing"]["primary"]["conditional"], base["conditional"]["all_services"],
          base["conditional"]["binary_gated_all_services"]]),
    ]
    fig, axes = plt.subplots(2, 1, figsize=(6.2, 5.6), sharey=True)
    for ax, (title, labels, fixed_stats, dense_stats) in zip(axes, panels):
        x = np.arange(len(labels)); width = .38
        for offset, stats, color, name in ((-width / 2, fixed_stats, BLUE, "Fixed-bag"),
                                           (width / 2, dense_stats, ORANGE, "Dense indexed")):
            means = [s["aggregate"]["macro_f1"]["mean"] if "aggregate" in s else s["macro_f1"]["mean"] for s in stats]
            errors = [s["aggregate"]["macro_f1"]["sample_sd"] if "aggregate" in s else s["macro_f1"]["sample_sd"] for s in stats]
            bars = ax.bar(x + offset, means, width, yerr=errors, capsize=2, color=color,
                          edgecolor="black", linewidth=.4, label=name)
            ax.bar_label(bars, labels=[f"{m:.3f}" for m in means], padding=2, fontsize=8)
        ax.set_xticks(x, labels, fontsize=8.5); ax.set_ylim(0, .88); ax.set_title(title, fontsize=9.5)
    for ax in axes:
        ax.set_ylabel("Response Macro-F1", fontsize=9)
    axes[0].legend(loc="upper left", frameon=False, fontsize=8.5)
    fig.tight_layout(); save(fig, "fig_routing_comparison.pdf", preview)


def response_vector_distribution(package, preview):
    fixed = package["phase1_fixed_bag_dispatch"]
    predicted = fixed["official_raw_summary"]["aggregate"]["prediction_distributions"]["dispatch_prediction_distribution"]
    truth = {}
    for row in fixed["per_video_records"][CASE_RUN]:
        key = ",".join(str(v) for v in row["dispatch_target"])
        truth[key] = truth.get(key, 0) + 1
    keys = [k for k in VECTOR_LABELS if k in truth or k in predicted]
    if (set(truth) | set(predicted)) - set(keys):
        raise ValueError("Unexpected response vectors")
    x = np.arange(len(keys)); width = .38
    true_counts = [truth.get(k, 0) for k in keys]
    pred_means = [predicted[k]["mean_count"] if k in predicted else 0 for k in keys]
    pred_errors = [predicted[k]["sample_std_count"] if k in predicted else 0 for k in keys]
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    bars_true = ax.bar(x - width / 2, true_counts, width, color="#BDBDBD", edgecolor="black",
                       linewidth=.4, label="Ground truth")
    bars_pred = ax.bar(x + width / 2, pred_means, width, yerr=pred_errors, capsize=2, color=BLUE,
                       edgecolor="black", linewidth=.4, label="Predicted (raw, mean of three seeds)")
    ax.bar_label(bars_true, labels=[str(c) for c in true_counts], padding=2, fontsize=8.5)
    ax.bar_label(bars_pred, labels=[f"{m:.0f}" for m in pred_means], padding=2, fontsize=8.5)
    ax.set_xticks(x, [VECTOR_LABELS[k] for k in keys])
    ax.set(xlabel="Response vector (P = Police, M = Medical, F = Fire)", ylabel="Test videos")
    ax.legend(frameon=False, fontsize=8.5); ax.set_ylim(0, max(max(true_counts), max(pred_means)) * 1.18)
    save(fig, "fig_response_vector_distribution.pdf", preview)


def temporal_category(package, preview):
    temporal = package["phase5_temporal_system"]
    stats = temporal["three_checkpoint_statistics"]["category_temporal"]
    whole = temporal["temporal_baselines"]["whole_video"]["by_category"]
    categories = sorted((c for c in stats if "ap_mean" in stats[c]), key=lambda c: stats[c]["ap_mean"]["mean"])
    means = [stats[c]["ap_mean"]["mean"] for c in categories]
    errors = [stats[c]["ap_mean"]["sample_sd"] for c in categories]
    labels = [f"{c} (n={int(stats[c]['videos']['mean'])})" for c in categories]
    fig, ax = plt.subplots(figsize=(5.1, 4.8)); y = np.arange(len(categories))
    ax.barh(y, means, xerr=errors, capsize=2, color=BLUE, edgecolor="black", linewidth=.35,
            label="Learned attention (mean ± SD, three seeds)")
    ax.scatter([whole[c]["ap_mean"] for c in categories], y, marker="|", s=160, linewidths=2, color=ORANGE,
               zorder=3, label="Whole-video reference (event prevalence)")
    ax.set_yticks(y, labels); ax.set_xlim(0, 1); ax.set_xlabel("Temporal average precision")
    ax.legend(loc="upper center", bbox_to_anchor=(.4, -.12), frameon=False, fontsize=8)
    save(fig, "fig_temporal_ap_by_category.pdf", preview)


def cad_example(package, preview):
    """Same content as the Phase-6 CAD timeline, drawn at text size for the thesis page width."""
    example = package["phase5_temporal_system"]["qualitative_cad_example"]
    fps = example["metadata"]["src_fps"]; duration = example["metadata"]["src_frame_count"] / fps
    fig, (ax, side) = plt.subplots(1, 2, figsize=(6.2, 3.0), gridspec_kw={"width_ratios": [2.3, 1]})
    names = ("Ground truth", "Primary prediction", "Display-gated alert")
    keys = ("ground_truth", "primary_ungated", "display_gated")
    styles = (("#D9D9D9", ".."), ("#666666", "//"), (ORANGE, ""))
    for y, (key, (color, hatch)) in enumerate(zip(keys, styles)):
        for start, end in example["intervals"][key]:
            ax.broken_barh([(start / fps, (end - start) / fps)], (y - .3, .6), facecolors=color,
                           edgecolors="black", linewidth=.5, hatch=hatch)
    ax.set_yticks(range(3), names, fontsize=8.5); ax.set_xlim(0, duration)
    ax.set_xlabel("Video time (s)", fontsize=9)
    ax.set_title(f"Offline temporal demonstration\n{example['video_id']}", fontsize=9.5)
    routing = example["routing"]; side.axis("off")
    text = (f"Binary score\n{routing['binary_score']:.3f}\n\n"
            f"Priority\n{routing['cad_priority']}\n\n"
            f"Services\n{', '.join(c.title() for c in routing['cad_services'])}\n\n"
            f"Alert window\n{routing['cad_alert']['anomaly_window']['start_sec']:.2f}\u2013"
            f"{routing['cad_alert']['anomaly_window']['end_sec']:.2f} s")
    side.text(.04, .98, text, va="top", fontsize=8.5, linespacing=1.1)
    side.text(.04, 0, "Offline demonstration;\nnot deployment evidence", va="bottom", fontsize=8, color="#444444")
    fig.tight_layout(); save(fig, "fig_temporal_cad_example.pdf", preview)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview-dir", type=Path)
    parser.add_argument("--select-only", action="store_true", help="Print the qualitative case selection and exit.")
    args = parser.parse_args(); configure()
    package = json.loads(PACKAGE.read_text(encoding="utf-8"))
    if args.select_only:
        for label, record in select_cases(load_case_records(package)):
            print(label, record["video_id"])
        return
    OUTPUT.mkdir(parents=True, exist_ok=False)
    cases = qualitative_cases(package, args.preview_dir)
    routing_comparison(package, args.preview_dir)
    response_vector_distribution(package, args.preview_dir)
    temporal_category(package, args.preview_dir)
    cad_example(package, args.preview_dir)
    print("PASS: revision figures in", OUTPUT, "| cases:", ", ".join(cases))


if __name__ == "__main__":
    main()
