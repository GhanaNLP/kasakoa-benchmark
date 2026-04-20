"""
reporting.py
Generates per-language and overall performance charts.
"""

import os
import re
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _model_label(filename):
    """Strip lang prefix and .csv suffix, convert underscores to slashes."""
    name = re.sub(r'^[a-z]+_', '', filename, count=1)
    name = name.replace('.csv', '')
    # Reverse the safe_id substitution (underscores back to slashes where appropriate)
    # Heuristic: known org prefixes
    for org in ['meta', 'google', 'nvidia', 'mistralai', 'deepseek-ai',
                'moonshotai', 'qwen', 'minimaxai', 'openai']:
        name = name.replace(f'{org}_', f'{org}/', 1)
    return name


def _load_lang_results(lang_code, output_dir):
    """Returns DataFrame with columns: model, mean_score, std_score, n"""
    rows = []
    for fname in sorted(os.listdir(output_dir)):
        if not fname.endswith('.csv'):
            continue
        fpath = os.path.join(output_dir, fname)
        try:
            df = pd.read_csv(fpath)
        except Exception:
            continue
        if 'similarity_score' not in df.columns:
            continue
        scores = df['similarity_score'].dropna()
        if len(scores) == 0:
            continue
        rows.append({
            'model': _model_label(fname),
            'mean_score': scores.mean(),
            'std_score': scores.std(),
            'n': len(scores),
        })
    return pd.DataFrame(rows)


# ─── Per-language charts ───────────────────────────────────────────────────────

def _chart_performance(results_df, lang_name, lang_code, out_path):
    df = results_df.sort_values('mean_score', ascending=True)
    n = len(df)
    fig, ax = plt.subplots(figsize=(10, max(4, n * 0.55)))

    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.85, n))
    bars = ax.barh(df['model'], df['mean_score'], xerr=df['std_score'],
                   color=colors, edgecolor='white', linewidth=0.5,
                   error_kw=dict(ecolor='#555555', capsize=3, linewidth=1))

    for bar, val in zip(bars, df['mean_score']):
        ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                f'{val:.3f}', va='center', ha='left', fontsize=8)

    ax.set_xlim(0, 1.05)
    ax.set_xlabel('Mean Cosine Similarity (BGE-M3)', fontsize=10)
    ax.set_title(f'{lang_name} — Model Performance on QA Benchmark', fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {out_path}")


def _chart_quadrant(results_df, lang_name, lang_code, out_path):
    """Performance vs consistency (lower std = more consistent)."""
    df = results_df.copy()
    if len(df) < 2:
        return

    fig, ax = plt.subplots(figsize=(9, 7))

    sc = ax.scatter(df['mean_score'], df['std_score'],
                    c=df['mean_score'], cmap='RdYlGn',
                    s=120, edgecolors='white', linewidths=0.8, zorder=3)

    med_x = df['mean_score'].median()
    med_y = df['std_score'].median()
    ax.axvline(med_x, color='#999999', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.axhline(med_y, color='#999999', linestyle='--', linewidth=0.8, alpha=0.7)

    for _, row in df.iterrows():
        ax.annotate(row['model'], (row['mean_score'], row['std_score']),
                    textcoords='offset points', xytext=(6, 4), fontsize=7.5)

    ax.set_xlabel('Mean Similarity Score (higher = better)', fontsize=10)
    ax.set_ylabel('Std Deviation (lower = more consistent)', fontsize=10)
    ax.set_title(f'{lang_name} — Performance vs Consistency', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Quadrant labels
    ax.text(med_x + 0.002, df['std_score'].max() * 0.96, 'High perf\ninconsistent',
            fontsize=7.5, color='#888888', ha='left')
    ax.text(df['mean_score'].min(), df['std_score'].max() * 0.96, 'Low perf\ninconsistent',
            fontsize=7.5, color='#888888', ha='left')
    ax.text(med_x + 0.002, df['std_score'].min() + 0.002, 'High perf\nconsistent',
            fontsize=7.5, color='#888888', ha='left')
    ax.text(df['mean_score'].min(), df['std_score'].min() + 0.002, 'Low perf\nconsistent',
            fontsize=7.5, color='#888888', ha='left')

    plt.colorbar(sc, ax=ax, label='Mean Score', shrink=0.7)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {out_path}")


def _save_summary_csv(results_df, lang_code, reports_dir):
    path = os.path.join(reports_dir, 'summary.csv')
    results_df.to_csv(path, index=False)
    print(f"    Saved: {path}")


# ─── Public API ───────────────────────────────────────────────────────────────

def generate_reports(lang_code, lang_name, output_dir, reports_dir):
    results = _load_lang_results(lang_code, output_dir)
    if results.empty:
        print("  No scored results found — skipping report generation.")
        return

    _chart_performance(results, lang_name, lang_code,
                       os.path.join(reports_dir, 'performance_comparison.png'))
    _chart_quadrant(results, lang_name, lang_code,
                    os.path.join(reports_dir, 'performance_vs_consistency_quadrant.png'))
    _save_summary_csv(results, lang_code, reports_dir)

    # Also regenerate the cross-language overall report if multiple langs exist
    _maybe_generate_overall(os.path.dirname(reports_dir), reports_dir)


def _maybe_generate_overall(reports_root, current_lang_dir):
    """
    If multiple language summary CSVs exist, generate overall charts
    in reports_root/.
    """
    all_rows = []
    for entry in os.scandir(reports_root):
        if not entry.is_dir():
            continue
        summary = os.path.join(entry.path, 'summary.csv')
        if not os.path.exists(summary):
            continue
        df = pd.read_csv(summary)
        df['lang'] = entry.name
        all_rows.append(df)

    if not all_rows or len(all_rows) < 2:
        return  # not enough data for meaningful cross-lang chart

    combined = pd.concat(all_rows, ignore_index=True)

    # Overall model performance (mean across all langs)
    model_perf = (
        combined.groupby('model')['mean_score']
        .mean()
        .sort_values(ascending=True)
    )

    fig, ax = plt.subplots(figsize=(10, max(4, len(model_perf) * 0.55)))
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.85, len(model_perf)))
    bars = ax.barh(model_perf.index, model_perf.values, color=colors,
                   edgecolor='white', linewidth=0.5)
    for bar, val in zip(bars, model_perf.values):
        ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                f'{val:.3f}', va='center', ha='left', fontsize=8)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel('Mean Cosine Similarity across Languages', fontsize=10)
    ax.set_title('Overall Model Performance — All Languages', fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(reports_root, 'model_performance.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Language difficulty (mean score per language)
    lang_perf = (
        combined.groupby('lang')['mean_score']
        .mean()
        .sort_values(ascending=True)
    )
    fig, ax = plt.subplots(figsize=(10, max(4, len(lang_perf) * 0.55)))
    colors2 = plt.cm.RdYlGn(np.linspace(0.2, 0.85, len(lang_perf)))
    bars2 = ax.barh(lang_perf.index, lang_perf.values, color=colors2,
                    edgecolor='white', linewidth=0.5)
    for bar, val in zip(bars2, lang_perf.values):
        ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                f'{val:.3f}', va='center', ha='left', fontsize=8)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel('Mean Cosine Similarity', fontsize=10)
    ax.set_title('Language Performance — Avg Across All Models', fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(reports_root, 'language_performance.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Overall charts updated in {reports_root}/")
