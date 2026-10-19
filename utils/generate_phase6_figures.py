"""Generate all Phase-6 thesis figures from the canonical package only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np
from sklearn.metrics import roc_curve

PACKAGE = Path("artifacts/stage2/evidence/canonical_thesis_v1/package.json")
OUTPUT = Path("artifacts/thesis_figures/phase6")
RUNS = ("dispatch_repro_s1337", "dispatch_repro_s2027", "dispatch_repro_s3407")
RUN_LABELS = ("Seed 1337", "Seed 2027", "Seed 3407")
CHANNELS = ("police", "medical", "fire")
BLUE, ORANGE, GREEN, PURPLE, GRAY = "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#666666"
COLORS = (BLUE, ORANGE, GREEN, PURPLE, GRAY)


def configure() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10,
                         "axes.labelsize": 9, "legend.fontsize": 8, "xtick.labelsize": 8,
                         "ytick.labelsize": 8, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.dpi": 160, "savefig.dpi": 300, "pdf.fonttype": 42,
                         "axes.grid": True, "grid.alpha": 0.22, "grid.linewidth": 0.6})


def save(fig: plt.Figure, name: str, preview: Path | None) -> None:
    fig.savefig(OUTPUT / name, format="pdf", bbox_inches="tight",
                metadata={"Title": name.removesuffix(".pdf"), "Creator": "Phase 6 canonical evidence generator"})
    if preview:
        preview.mkdir(parents=True, exist_ok=True)
        fig.savefig(preview / name.replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)


def error_bars(ax, labels, means, errors, title, ylabel="Score", ylim=(0, 1), colors=None):
    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=errors, capsize=3, color=colors or [BLUE] * len(labels),
                  edgecolor="black", linewidth=0.5)
    ax.set_xticks(x, labels)
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.bar_label(bars, labels=[f"{value:.3f}" for value in means], padding=3, fontsize=7)


def architecture(package, preview):
    stages = package["figure_context"]["pipeline_stages"]
    labels = ("Indexed\nvideo", "Deterministic\nclips", "LoRA-adapted\nfrozen\nVideoMAE\nfeatures",
              "Normal-\nprototype\ndistance", "Attention\nMIL binary +\nresponse\nheads",
              "Binary\ngate", "Offline\nCAD-style\nalert")
    if list(stages) != [label.replace("-\n", "-").replace("\n", " ") for label in labels]:
        raise ValueError("Pipeline labels no longer match the canonical stage sequence")
    fig, ax = plt.subplots(figsize=(6.9, 2.3))
    ax.set_position([0, 0, 1, 1]); ax.axis("off")
    left, right, gap = .005, .995, .02
    box_width = (right - left - gap * (len(labels) - 1)) / len(labels)
    bottom, height = .25, .46
    for i, label in enumerate(labels):
        x0 = left + i * (box_width + gap)
        if i:
            previous_right = x0 - gap
            ax.add_patch(FancyArrowPatch((previous_right + .002, bottom + height / 2),
                                         (x0 - .002, bottom + height / 2),
                                         arrowstyle="-|>", mutation_scale=8,
                                         color="#444444", linewidth=1.0, zorder=1))
        ax.add_patch(Rectangle((x0, bottom), box_width, height, facecolor="white",
                               edgecolor="#333333", linewidth=.9, zorder=2))
        ax.text(x0 + box_width / 2, bottom + height / 2, label, ha="center", va="center",
                fontsize=8, linespacing=1.15, zorder=3)
    ax.text(.5, .88, "Offline video-anomaly and response-routing pipeline",
            ha="center", va="center", fontsize=10)
    ax.text(.5, .06, "Frozen features and prototype distances enter the two-head MIL model;\nthe binary decision gates response routing.",
            ha="center", color="#444444", fontsize=8)
    save(fig, "fig_architecture_dataflow.pdf", preview)


def provenance(package, preview):
    fig, ax = plt.subplots(figsize=(6.9, 2.3)); ax.set_position([0, 0, 1, 1]); ax.axis("off")
    labels = ["Fixed\ntraining\ninputs", "Three seeds /\ncheckpoints", "Validation-only\nselection +\nablation",
              "Frozen\ndecisions and\nprotocol", "Test\nevaluation of\nfixed\ncheckpoints", "Immutable\nevidence +\nSHA-256\nhashes"]
    bands = ["TRAIN", "TRAIN", "VALIDATION", "FREEZE", "TEST", "OUTPUT"]
    xs = np.linspace(.078, .922, len(labels)); half = .072
    for index, (x, label, band) in enumerate(zip(xs, labels, bands)):
        edge = BLUE if band == "TRAIN" else ORANGE if band == "VALIDATION" else GREEN if band == "TEST" else "#444444"
        ax.add_patch(FancyBboxPatch((x-half, .25), 2*half, .44, boxstyle="round,pad=.006,rounding_size=.003",
                                    facecolor="white", edgecolor=edge, linewidth=.9))
        ax.text(x, .47, label, ha="center", va="center", fontsize=8)
        ax.text(x, .75, band, ha="center", va="center", fontsize=7.5, color=edge, weight="bold")
        if index < len(labels)-1:
            ax.add_patch(FancyArrowPatch((x+half+.002, .47), (xs[index+1]-half-.002, .47),
                                         arrowstyle="-|>", mutation_scale=7, color="#444444", linewidth=.8))
    ax.text(.5, .88, "Evaluation protocol and provenance sequence", ha="center", fontsize=10)
    ax.text(.5, .06, "Validation does not generate test evidence; it freezes checkpoint identities\nand evaluation decisions before test execution.",
            ha="center", color="#444444", fontsize=8)
    save(fig, "fig_protocol_provenance.pdf", preview)


def headline(package, preview):
    fixed = package["phase1_fixed_bag_dispatch"]["official_raw_summary"]["aggregate"]["metrics"]
    indexed = package["phase5_temporal_system"]["three_checkpoint_statistics"]
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.6))
    error_bars(axes[0], ["Fixed-bag\nAUROC", "Indexed\nAUROC", "Fixed-bag\nBinary F1", "Indexed\nBinary F1"],
               [fixed["binary_auroc"]["mean"], indexed["binary"]["auroc"]["mean"],
                fixed["binary_f1"]["mean"], indexed["binary"]["f1"]["mean"]],
               [fixed["binary_auroc"]["sample_std"], indexed["binary"]["auroc"]["sample_sd"],
                fixed["binary_f1"]["sample_std"], indexed["binary"]["f1"]["sample_sd"]],
               "Binary anomaly detection (test, n=3)", colors=[BLUE, GREEN, BLUE, GREEN])
    routing = indexed["routing"]["primary"]
    error_bars(axes[1], ["Fixed-bag\nraw", "Indexed\nraw", "Indexed\ngated", "Indexed\nconditional"],
               [fixed["macro_f1"]["mean"], routing["raw"]["macro_f1"]["mean"],
                routing["gated"]["macro_f1"]["mean"], routing["conditional"]["macro_f1"]["mean"]],
               [fixed["macro_f1"]["sample_std"], routing["raw"]["macro_f1"]["sample_sd"],
                routing["gated"]["macro_f1"]["sample_sd"], routing["conditional"]["macro_f1"]["sample_sd"]],
               "Response-routing Macro-F1 (test, n=3)", colors=[BLUE, GREEN, ORANGE, PURPLE])
    fig.suptitle("Official three-checkpoint headline results (mean ± sample SD)", weight="bold")
    fig.tight_layout(); save(fig, "fig_headline_metrics.pdf", preview)


def binary_roc(package, preview):
    scores = package["phase5_temporal_system"]["binary_scores"]
    fig, ax = plt.subplots(figsize=(5.4, 4.3))
    for tag, label, color in zip(RUNS, RUN_LABELS, COLORS):
        fpr, tpr, _ = roc_curve(scores[tag]["binary_true"], scores[tag]["binary_score"])
        auc = package["phase5_temporal_system"]["auroc_verification"][tag]["canonical"]
        ax.plot(fpr, tpr, lw=1.8, color=color, label=f"{label} (AUROC {auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color=GRAY, lw=1, label="Chance")
    ax.set(xlabel="False-positive rate", ylabel="True-positive rate", xlim=(0, 1), ylim=(0, 1.01),
           title="Dense indexed binary anomaly ROC (test, 290 videos)")
    ax.legend(loc="lower right", frameon=True); ax.set_aspect("equal", adjustable="box")
    save(fig, "fig_binary_roc.pdf", preview)


def pr_curves(package, preview):
    curves = package["phase1_fixed_bag_dispatch"]["per_channel_precision_recall_curves"]
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.75), sharex=True, sharey=True)
    line_styles = (("#222222", "-"), (BLUE, "--"), (ORANGE, ":"))
    for ax, channel in zip(axes, CHANNELS):
        for tag, label, (color, style) in zip(RUNS, RUN_LABELS, line_styles):
            curve = curves[tag][channel]
            recall = np.asarray(curve["recall"], dtype=float)
            precision = np.asarray(curve["precision"], dtype=float)
            thresholds = np.asarray(curve["thresholds"], dtype=float)
            if len(recall) != len(precision) or len(thresholds) + 1 != len(recall):
                raise ValueError(f"Malformed PR arrays for {tag}/{channel}")
            if recall[0] != 0.0 or precision[0] != 1.0 or np.any(np.diff(recall) < 0):
                raise ValueError(f"PR points are not in canonical threshold order for {tag}/{channel}")
            if np.any(np.diff(thresholds) >= 0):
                raise ValueError(f"PR thresholds are not strictly descending for {tag}/{channel}")
            ax.step(recall, precision, where="pre", color=color, linestyle=style, lw=1.55,
                    label=f"{label} (AP {curve['average_precision']:.3f})")
        ax.set_title(channel.title()); ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="Recall")
        ax.legend(loc="upper center", bbox_to_anchor=(.5, -.24), frameon=False,
                  fontsize=7.2, handlelength=2.4, labelspacing=.35)
    axes[0].set_ylabel("Precision")
    fig.suptitle("Fixed-bag response precision–recall curves (test, 290 videos)")
    fig.tight_layout(rect=(0, .10, 1, 1)); save(fig, "fig_per_channel_pr_curves.pdf", preview)


def per_channel(package, preview):
    stats = package["phase5_temporal_system"]["three_checkpoint_statistics"]["routing"]["primary"]["gated"]["channels"]
    fig, ax = plt.subplots(figsize=(7.4, 4.0)); x = np.arange(3); width = .24
    for offset, metric, color in zip((-width, 0, width), ("precision", "recall", "f1"), (BLUE, ORANGE, GREEN)):
        means = [stats[c][metric]["mean"] for c in CHANNELS]
        errors = [stats[c][metric]["sample_sd"] for c in CHANNELS]
        ax.bar(x + offset, means, width, yerr=errors, capsize=2.5, label=metric.title(), color=color,
               edgecolor="black", linewidth=.4)
    ax.set_xticks(x, [c.title() for c in CHANNELS]); ax.set_ylim(0, 1); ax.set_ylabel("Score")
    ax.set_title("Dense indexed gated routing by response channel\nTest, 290 videos; mean ± sample SD across three checkpoints")
    ax.legend(ncols=3, loc="upper center", frameon=True)
    save(fig, "fig_per_channel_prf1.pdf", preview)


def joint_distribution(package, preview):
    distribution = package["phase1_fixed_bag_dispatch"]["official_raw_summary"]["aggregate"]["prediction_distributions"]["dispatch_prediction_distribution"]
    keys = sorted(distribution)
    service_labels = {"1,0,0": "Police", "1,0,1": "Police +\nFire",
                      "1,1,0": "Police +\nMedical", "1,1,1": "Police + Medical\n+ Fire"}
    if set(keys) != set(service_labels):
        raise ValueError("Unexpected joint dispatch vectors")
    labels = [service_labels[key] for key in keys]
    means = [distribution[key]["mean_count"] for key in keys]
    errors = [distribution[key]["sample_std_count"] for key in keys]
    fig, ax = plt.subplots(figsize=(6.8, 4.0)); bars = ax.bar(labels, means, yerr=errors, capsize=3,
        color=["#D9D9D9", "#BDBDBD", "#969696", "#636363"],
        edgecolor="#222222", linewidth=.6)
    ax.bar_label(bars, labels=[f"{m:.1f}" for m in means], padding=3, fontsize=8)
    ax.set(ylabel="Videos per checkpoint", xlabel="Predicted service set",
           title="Raw fixed-bag joint dispatch predictions\nTest, 290 videos; mean ± sample SD across three seeds")
    ax.set_ylim(bottom=0)
    save(fig, "fig_joint_dispatch_distribution.pdf", preview)


def routing_comparison(package, preview):
    fixed = package["phase1_fixed_bag_dispatch"]["derived_matched_conditions"]
    indexed = package["phase5_temporal_system"]["three_checkpoint_statistics"]
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.6), sharey=True, gridspec_kw={"width_ratios": [3, 2]})
    all_labels = ["Fixed-bag\nraw", "Fixed-bag\ngated", "Indexed\nraw", "Indexed\ngated",
                  "All\nservices", "Gated all\nservices"]
    all_stats = [fixed["raw_all_290"]["aggregate"]["macro_f1"], fixed["binary_gated_all_290"]["aggregate"]["macro_f1"],
                 indexed["routing"]["primary"]["raw"]["macro_f1"], indexed["routing"]["primary"]["gated"]["macro_f1"],
                 indexed["routing_baselines"]["all"]["all_services"]["macro_f1"],
                 indexed["routing_baselines"]["all"]["binary_gated_all_services"]["macro_f1"]]
    cond_labels = ["Fixed-bag\nraw", "Indexed\nraw", "All\nservices", "Indexed\ngated all\nservices"]
    cond_stats = [fixed["conditional_anomaly_140"]["aggregate"]["macro_f1"],
                  indexed["routing"]["primary"]["conditional"]["macro_f1"],
                  indexed["routing_baselines"]["conditional"]["all_services"]["macro_f1"],
                  indexed["routing_baselines"]["conditional"]["binary_gated_all_services"]["macro_f1"]]
    for ax, labels, values, title in ((axes[0], all_labels, all_stats, "All test videos (n=290)"),
                                      (axes[1], cond_labels, cond_stats, "Anomalous videos only (n=140)")):
        means = [v["mean"] for v in values]; errors = [v["sample_sd"] for v in values]
        colors = (["#777777", "#AAAAAA", BLUE, "#79A9C9", "#D0D0D0", "#969696"]
                  if len(labels) == 6 else ["#777777", BLUE, "#D0D0D0", "#969696"])
        bars = ax.bar(np.arange(len(labels)), means, yerr=errors, capsize=2.5,
                      color=colors,
                      edgecolor="black", linewidth=.4)
        ax.set_xticks(np.arange(len(labels)), labels); ax.set_ylim(0, .85); ax.set_title(title, fontsize=9)
        ax.tick_params(axis="x", labelsize=7.5)
        ax.bar_label(bars, labels=[f"{m:.3f}" for m in means], padding=2, fontsize=7.5)
    axes[0].set_ylabel("Macro-F1")
    fig.suptitle("Matched response-routing conditions and trivial baselines\nMean ± sample SD across three checkpoints")
    fig.tight_layout(); save(fig, "fig_routing_comparison.pdf", preview)


def teacher_ablation(package, preview):
    agg = package["phase3_teacher_ablation"]["aggregates"]
    keys = ("full", "no_input", "uniform", "neither"); labels = ("Full", "No score input", "Uniform sampling", "Neither")
    means = [agg[k]["macro_f1"]["mean"] for k in keys]; errors = [agg[k]["macro_f1"]["sample_std"] for k in keys]
    fig, ax = plt.subplots(figsize=(6.3, 3.8)); x = np.arange(len(labels))
    ax.errorbar(x, means, yerr=errors, fmt="o", color="#333333", markerfacecolor=BLUE,
                markeredgecolor="#222222", markersize=6, elinewidth=1.1, capsize=4, zorder=3)
    for xpos, mean in zip(x, means):
        ax.annotate(f"{mean:.3f}", (xpos, mean), xytext=(0, 9), textcoords="offset points",
                    ha="center", fontsize=8)
    ax.set_xticks(x, labels)
    ax.set_ylim(.40, .48); ax.set_ylabel("Macro-F1")
    ax.set_title("Teacher-mechanism ablation (validation only, n=161)\nMean ± sample SD across three training seeds")
    save(fig, "fig_teacher_ablation.pdf", preview)


def historical(package, preview):
    section = package["phase2_historical_14_class"]
    class_names = section["task"]["class_names"]
    abbreviations = ("Nrm", "Abu", "Arr", "Ars", "Asl", "Bur", "Exp",
                     "Fgt", "RdA", "Rob", "Sht", "Shp", "Stl", "Van")
    names = ("hier_cascaded_v3", "hier_lora_e2e")
    fig, axes = plt.subplots(1, 2, figsize=(11.3, 5.25), sharey=True)
    image = None
    for ax, name in zip(axes, names):
        item = section["experiments"][name]
        if item["test_confusion"]["class_names"] != class_names:
            raise ValueError(f"Historical class order mismatch for {name}")
        matrix = np.asarray(item["test_confusion"]["matrix"], dtype=float)
        normalized = np.divide(matrix, matrix.sum(axis=1, keepdims=True), out=np.zeros_like(matrix), where=matrix.sum(axis=1, keepdims=True) != 0)
        image = ax.imshow(normalized, cmap="cividis", vmin=0, vmax=1, aspect="auto")
        ax.set_title(f"{name.replace('_', ' ')}\nMacro-F1 {item['external_test']['macro_f1']:.3f}")
        ax.set(xlabel="Predicted class", ylabel="True class", xticks=range(14), yticks=range(14))
        ax.set_xticklabels(abbreviations, rotation=55, ha="right")
        ax.set_yticklabels(class_names)
        ax.grid(False)
    fig.colorbar(image, ax=axes, fraction=.025, pad=.03, label="Row-normalized fraction")
    fig.subplots_adjust(top=.78, bottom=.16, left=.13, right=.91, wspace=.18)
    fig.suptitle("Verified historical 14-class test confusion matrices", y=.98, weight="bold")
    fig.text(.5, .90, "Non-controlled historical outputs; abbreviated x labels follow the frozen class order shown on the y axis",
             ha="center", fontsize=9, color=GRAY)
    save(fig, "fig_historical_14class.pdf", preview)


def temporal_category(package, preview):
    stats = package["phase5_temporal_system"]["three_checkpoint_statistics"]["category_temporal"]
    categories = sorted((c for c in stats if "ap_mean" in stats[c]),
                        key=lambda c: stats[c]["ap_mean"]["mean"])
    means = [stats[c]["ap_mean"]["mean"] for c in categories]; errors = [stats[c]["ap_mean"]["sample_sd"] for c in categories]
    sizes = [int(stats[c]["videos"]["mean"]) for c in categories]
    labels = [f"{c} (n={n})" for c, n in zip(categories, sizes)]
    fig, ax = plt.subplots(figsize=(5.6, 4.6)); y = np.arange(len(categories))
    ax.barh(y, means, xerr=errors, capsize=2, color=BLUE, edgecolor="black", linewidth=.35)
    ax.set_yticks(y, labels); ax.set_xlim(0, 1); ax.set_xlabel("Temporal average precision")
    ax.set_title("Dense indexed temporal AP by anomaly category\nTest; mean ± sample SD across three checkpoints")
    save(fig, "fig_temporal_ap_by_category.pdf", preview)


def temporal_iou(package, preview):
    section = package["phase5_temporal_system"]; stats = section["three_checkpoint_statistics"]["temporal"]["iou_at_least"]
    thresholds = ("0.1", "0.3", "0.5", "0.7")
    model = [stats[t]["mean"] for t in thresholds]; errors = [stats[t]["sample_sd"] for t in thresholds]
    whole = [section["temporal_baselines"]["whole_video"]["overall"]["iou_at_least"][t] for t in thresholds]
    random = [section["temporal_baselines"]["random_clips"]["overall"]["iou_at_least"][t] for t in thresholds]
    fig, ax = plt.subplots(figsize=(7.2, 4)); x = np.arange(4); w = .25
    ax.bar(x-w, model, w, yerr=errors, capsize=2.5, label="Model (n=3 checkpoints)", color=BLUE, edgecolor="black", linewidth=.4)
    ax.bar(x, whole, w, label="Whole-video baseline (1 replicate)", color=ORANGE, hatch="//", edgecolor="black", linewidth=.4)
    ax.bar(x+w, random, w, label="Random-clips baseline (1 frozen replicate)", color=GREEN, hatch="..", edgecolor="black", linewidth=.4)
    ax.set_xticks(x, [f"IoU ≥ {t}" for t in thresholds]); ax.set_ylim(0, .75); ax.set_ylabel("Fraction of 135 eligible videos")
    ax.set_title("Fixed-rule temporal overlap thresholds (test)"); ax.legend(frameon=True)
    save(fig, "fig_temporal_iou_thresholds.pdf", preview)


def temporal_baselines(package, preview):
    section = package["phase5_temporal_system"]; stats = section["three_checkpoint_statistics"]["temporal"]
    names = ["Model", "Whole video", "Uniform clips", "Random clips", "Teacher distance"]
    baselines = section["temporal_baselines"]
    ap = [stats["ap_mean"]["mean"]] + [baselines[k]["overall"]["ap_mean"] for k in ("whole_video", "uniform_clips", "random_clips", "teacher_distance")]
    iou = [stats["iou_mean"]["mean"]] + [baselines[k]["overall"]["iou_mean"] for k in ("whole_video", "uniform_clips", "random_clips", "teacher_distance")]
    if any(value is None for value in ap + iou):
        raise ValueError("Temporal baseline plot contains a missing value")
    uniform = baselines["uniform_clips"]["overall"]
    if uniform["iou_mean"] != 0.0 or uniform["zero_iou_count"] != uniform["videos"]:
        raise ValueError("Uniform-clips zero IoU is not supported by all eligible videos")
    ap_err = [stats["ap_mean"]["sample_sd"], 0, 0, 0, 0]; iou_err = [stats["iou_mean"]["sample_sd"], 0, 0, 0, 0]
    fig, ax = plt.subplots(figsize=(8.1, 4.35)); y = np.arange(5); h = .34
    ax.barh(y-h/2, ap, h, label="Temporal AP", color=BLUE, edgecolor="black", linewidth=.4)
    ax.barh(y+h/2, iou, h, label="Fixed-rule IoU", color="#9A9A9A",
            edgecolor="black", linewidth=.4)
    ax.errorbar(ap[0], y[0]-h/2, xerr=ap_err[0], fmt="none", ecolor="black",
                elinewidth=1.1, capsize=2.5, zorder=3)
    ax.errorbar(iou[0], y[0]+h/2, xerr=iou_err[0], fmt="none", ecolor="black",
                elinewidth=1.1, capsize=2.5, zorder=3)
    ax.annotate("0.000", (0, y[2]+h/2), xytext=(4, 0), textcoords="offset points",
                va="center", fontsize=7.5, color="#444444")
    ax.set_yticks(y, names); ax.invert_yaxis(); ax.set_xlim(0, .4); ax.set_xlabel("Score")
    ax.set_title("Temporal model and reference baselines (135 eligible test anomalies)\nError bars only for model: mean ± sample SD, n=3; baselines: one frozen replicate")
    ax.legend(frameon=True, loc="lower right")
    fig.tight_layout(); save(fig, "fig_temporal_baselines_comparison.pdf", preview)


def coverage(package, preview):
    data = package["phase5_temporal_system"]["coverage"]
    keys = ("dense_realized", "sparse", "fixed_bag"); labels = ("Dense indexed", "Sparse", "Fixed bag")
    values = [100 * data[key]["frame_weighted_coverage"] for key in keys]
    fig, ax = plt.subplots(figsize=(5.8, 3.8)); bars = ax.bar(labels, values, color=[GREEN, BLUE, ORANGE], edgecolor="black", linewidth=.5)
    ax.bar_label(bars, labels=[f"{value:.2f}%" for value in values], padding=3)
    ax.set_ylim(0, 108); ax.set_ylabel("Frame-weighted source coverage (%)")
    ax.set_title("Temporal coverage under matched source-frame geometry\n290 test videos; denominator 1,111,808 source frames")
    save(fig, "fig_coverage_comparison.pdf", preview)


def cad_example(package, preview):
    example = package["phase5_temporal_system"]["qualitative_cad_example"]
    fps = example["metadata"]["src_fps"]; duration = example["metadata"]["src_frame_count"] / fps
    fig, (ax, side) = plt.subplots(1, 2, figsize=(6.9, 2.9), gridspec_kw={"width_ratios": [2.3, 1]})
    names = ("Ground truth", "Primary prediction", "Display-gated alert")
    keys = ("ground_truth", "primary_ungated", "display_gated")
    styles = (("#D9D9D9", ".."), ("#666666", "//"), (ORANGE, ""))
    for y, (name, key, (color, hatch)) in enumerate(zip(names, keys, styles)):
        for start, end in example["intervals"][key]:
            collection = ax.broken_barh([(start/fps, (end-start)/fps)], (y-.3, .6), facecolors=color,
                                        edgecolors="black", linewidth=.5, hatch=hatch)
    ax.set_yticks(range(3), names); ax.set_xlim(0, duration); ax.set_xlabel("Video time (s)")
    ax.set_title(f"Offline temporal demonstration\n{example['video_id']}")
    routing = example["routing"]; side.axis("off")
    text = (f"Binary score\n{routing['binary_score']:.3f}\n\n"
            f"Priority\n{routing['cad_priority']}\n\n"
            f"Services\n{', '.join(c.title() for c in routing['cad_services'])}\n\n"
            f"Alert window\n{routing['cad_alert']['anomaly_window']['start_sec']:.2f}–"
            f"{routing['cad_alert']['anomaly_window']['end_sec']:.2f} s")
    side.text(.04, .98, text, va="top", fontsize=8, linespacing=1.1)
    side.text(.04, 0, "Offline demonstration;\nnot deployment evidence", va="bottom", fontsize=7.5, color="#444444")
    fig.tight_layout(); save(fig, "fig_temporal_cad_example.pdf", preview)


def stability(package, preview):
    value = package["phase1_fixed_bag_dispatch"]["prediction_stability"]
    overall_labels = ["Binary\ndecision", "Exact dispatch\nvector"]
    overall = [value["binary_prediction_unanimous_fraction"], value["dispatch_exact_vector_unanimous_fraction"]]
    channels = [value["dispatch_channel_unanimous_fraction"][c] for c in CHANNELS]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.75), sharey=True,
                             gridspec_kw={"width_ratios": [2, 3]})
    for ax, labels, means, title, color in (
            (axes[0], overall_labels, overall, "Overall outputs", BLUE),
            (axes[1], [c.title() for c in CHANNELS], channels, "Response channels", "#777777")):
        bars = ax.bar(labels, means, color=color, edgecolor="black", linewidth=.5, width=.68)
        ax.bar_label(bars, labels=[f"{100*m:.1f}%" for m in means], padding=3, fontsize=8)
        ax.set_ylim(0, 1.08); ax.set_title(title)
    axes[0].set_ylabel("Fraction with unanimous prediction")
    fig.suptitle("Fixed-bag prediction agreement across three training seeds\nTest, 290 videos")
    fig.tight_layout();
    save(fig, "fig_prediction_stability.pdf", preview)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview-dir", type=Path)
    parser.add_argument("--only", nargs="*", choices=("architecture", "provenance", "headline", "binary_roc",
                        "pr_curves", "per_channel", "joint_distribution", "routing_comparison",
                        "teacher_ablation", "historical", "temporal_category", "temporal_iou",
                        "temporal_baselines", "coverage", "cad_example", "stability"))
    parser.add_argument("--replace", action="store_true",
                        help="Replace only the explicitly selected canonical PDFs after archiving them.")
    args = parser.parse_args(); configure()
    if OUTPUT.exists() and not args.replace:
        raise FileExistsError(f"Refusing to overwrite figure directory: {OUTPUT}")
    if args.replace and not args.only:
        raise ValueError("--replace requires an explicit --only list")
    OUTPUT.mkdir(parents=True, exist_ok=args.replace)
    package = json.loads(PACKAGE.read_text(encoding="utf-8"))
    generators = {"architecture": architecture, "provenance": provenance, "headline": headline,
                  "binary_roc": binary_roc, "pr_curves": pr_curves, "per_channel": per_channel,
                  "joint_distribution": joint_distribution, "routing_comparison": routing_comparison,
                  "teacher_ablation": teacher_ablation, "historical": historical,
                  "temporal_category": temporal_category, "temporal_iou": temporal_iou,
                  "temporal_baselines": temporal_baselines, "coverage": coverage,
                  "cad_example": cad_example, "stability": stability}
    selected = args.only or list(generators)
    for name in selected:
        generators[name](package, args.preview_dir)
    print(f"PASS: generated {len(selected)} vector PDF figure(s) in {OUTPUT}")


if __name__ == "__main__":
    main()
