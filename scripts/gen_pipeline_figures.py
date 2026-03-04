#!/usr/bin/env python3
"""
Generate two minimalistic, monochrome figures:
  figures/chain_compression.png
  figures/nl_rendering.png

Run:
    python3 scripts/gen_pipeline_figures.py
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "figures")
os.makedirs(OUT_DIR, exist_ok=True)

BG    = "#FFFFFF"
LF    = "#F4F4F4"
BORD  = "#1A1A1A"
MID   = "#555555"
FAINT = "#AAAAAA"


def new_ax(w, h):
    fig = plt.figure(figsize=(w, h), dpi=150)
    fig.patch.set_facecolor(BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w); ax.set_ylim(0, h)
    ax.axis("off"); ax.set_facecolor(BG)
    return fig, ax


def rbox(ax, cx, cy, w, h, text="", fc=LF, ec=BORD, tc=BORD,
         fs=9, bold=False, mono=False, lw=1.0, z=2, ls="-"):
    patch = mpatches.FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle="round,pad=0.1", fc=fc, ec=ec,
        lw=lw, ls=ls, zorder=z)
    ax.add_patch(patch)
    if text:
        ff = "monospace" if mono else "DejaVu Sans"
        fw = "bold" if bold else "normal"
        ax.text(cx, cy, text, ha="center", va="center", fontsize=fs,
                color=tc, fontfamily=ff, fontweight=fw,
                multialignment="center", zorder=z+1, linespacing=1.6)


def arr(ax, x1, y1, x2, y2, lw=1.0, hw=0.15, hl=0.20, z=5):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), zorder=z,
                arrowprops=dict(
                    arrowstyle=f"->, head_width={hw}, head_length={hl}",
                    color=BORD, lw=lw))


def t(ax, x, y, s, fs=9, c=BORD, bold=False, italic=False,
      ha="center", va="center", mono=False, z=6):
    fw = "bold" if bold else "normal"
    st = "italic" if italic else "normal"
    ff = "monospace" if mono else "DejaVu Sans"
    ax.text(x, y, s, ha=ha, va=va, fontsize=fs, color=c,
            fontweight=fw, style=st, fontfamily=ff,
            multialignment=ha, zorder=z, linespacing=1.6)


def hline(ax, x1, x2, y, lw=0.7, ls=(0, (4, 3))):
    ax.plot([x1, x2], [y, y], color=FAINT, lw=lw, ls=ls, zorder=3)


# ════════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — CHAIN COMPRESSION
# ════════════════════════════════════════════════════════════════════════════════
def fig_compression():
    FW, FH = 13.5, 6.2
    fig, ax = new_ax(FW, FH)

    # Title
    t(ax, FW/2, FH - 0.38, "Chain Compression", fs=14, bold=True)
    t(ax, FW/2, FH - 0.82,
      "Consecutive steps are merged into one segment; "
      "intermediate formulas are absorbed and removed from the trace.",
      fs=9, c=MID, italic=True)

    # Geometry
    BW, BH = 4.8, 1.12
    BX   = 3.0        # left column centre-x
    RBX  = FW - BX    # right column centre-x  (= 10.5)
    S1Y  = 3.70       # segment 1 centre-y
    S2Y  = 1.90       # segment 2 centre-y
    ARRY = (S1Y + S2Y) / 2   # = 2.80

    ARRX1 = BX + BW/2 + 0.32    # = 5.72
    ARRX2 = RBX - BW/2 - 0.32   # = 7.78

    # ── Column headers (safely above the boxes) ────────────────────────────────
    # S1 box top = S1Y + BH/2 = 3.70 + 0.56 = 4.26
    # Headers go at y = 4.72 → gap of 0.46 ✓
    t(ax, BX,  4.72, "max_n = 1", fs=10.5, bold=True)
    t(ax, RBX, 4.72, "max_n = 2", fs=10.5, bold=True)

    # ── LEFT: two separate segments ────────────────────────────────────────────
    rbox(ax, BX, S1Y, BW, BH,
         "Segment 1\n──────────────────────\n"
         "inputs:   P→Q ,  P\n"
         "output:  Q")

    # Dashed connector showing Q passes between segments
    ax.plot([BX, BX], [S1Y - BH/2, S2Y + BH/2],
            color=BORD, lw=0.9, ls=(0, (2, 2)), zorder=4)

    # Q annotation (to the right of the connector)
    t(ax, BX + 0.18, ARRY + 0.06, "Q", fs=9.5, ha="left")
    t(ax, BX + 0.60, ARRY,
      "(intermediate:\n output of seg 1,\n input of seg 2)",
      fs=7.5, c=MID, italic=True, ha="left")

    rbox(ax, BX, S2Y, BW, BH,
         "Segment 2\n──────────────────────\n"
         "inputs:   Q→R ,  Q\n"
         "output:  R")

    # Left bracket
    BL = 0.30
    B1, B2 = S2Y - BH/2 - 0.10, S1Y + BH/2 + 0.10
    ax.plot([BL, BL], [B1, B2], color=FAINT, lw=1.4, zorder=3)
    ax.plot([BL, BL + 0.14], [B1, B1], color=FAINT, lw=1.4, zorder=3)
    ax.plot([BL, BL + 0.14], [B2, B2], color=FAINT, lw=1.4, zorder=3)

    # ── Center arrow ───────────────────────────────────────────────────────────
    arr(ax, ARRX1, ARRY, ARRX2, ARRY, lw=1.6, hw=0.22, hl=0.30)
    MCX = (ARRX1 + ARRX2) / 2
    t(ax, MCX, ARRY + 0.30, "compress", fs=9.5, bold=True)
    t(ax, MCX, ARRY - 0.25, "(max_n = 2)", fs=8.5, c=MID, italic=True)

    # ── RIGHT: one merged CompressedSegment ────────────────────────────────────
    MERGE_H = (S1Y + BH/2 + 0.12) - (S2Y - BH/2 - 0.12)   # ≈ 3.12
    MERGE_Y = (S1Y + S2Y) / 2                               # = 2.80
    # Merged box top = MERGE_Y + MERGE_H/2 ≈ 4.36
    # Column header at 4.72 → gap ≈ 0.36 ✓

    merged = mpatches.FancyBboxPatch(
        (RBX - BW/2, MERGE_Y - MERGE_H/2), BW, MERGE_H,
        boxstyle="round,pad=0.1", fc=LF, ec=BORD,
        lw=1.0, ls=(0, (6, 3)), zorder=2)
    ax.add_patch(merged)

    # Inputs row
    t(ax, RBX, MERGE_Y + MERGE_H/2 - 0.40,
      "inputs:   P→Q ,  P ,  Q→R", fs=9)

    # Divider
    hline(ax, RBX - BW/2 + 0.25, RBX + BW/2 - 0.25, MERGE_Y + 0.12)

    # Absorbed Q row (greyed, with strikethrough)
    ABS_Y = MERGE_Y - 0.10
    t(ax, RBX, ABS_Y,
      "Q   (absorbed — not in trace)", fs=8.5, c=FAINT, italic=True)
    # Strikethrough only over "Q"
    ax.plot([RBX - 1.85, RBX - 1.43], [ABS_Y, ABS_Y],
            color=FAINT, lw=0.9, zorder=7)

    # Divider
    hline(ax, RBX - BW/2 + 0.25, RBX + BW/2 - 0.25, MERGE_Y - 0.40)

    # Output row
    t(ax, RBX, MERGE_Y - MERGE_H/2 + 0.38, "output:  R", fs=9)

    # Right bracket
    BR = FW - 0.30
    M1, M2 = MERGE_Y - MERGE_H/2 - 0.10, MERGE_Y + MERGE_H/2 + 0.10
    ax.plot([BR, BR], [M1, M2], color=FAINT, lw=1.4, zorder=3)
    ax.plot([BR - 0.14, BR], [M1, M1], color=FAINT, lw=1.4, zorder=3)
    ax.plot([BR - 0.14, BR], [M2, M2], color=FAINT, lw=1.4, zorder=3)

    out = os.path.join(OUT_DIR, "chain_compression.png")
    plt.savefig(out, dpi=150, bbox_inches="tight",
                facecolor=BG, pad_inches=0.18)
    plt.close()
    print(f"Saved → {out}")


# ════════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — NL RENDERING → TRAINING JSONL
# ════════════════════════════════════════════════════════════════════════════════
def fig_rendering():
    FW, FH = 14.0, 8.0
    fig, ax = new_ax(FW, FH)

    # Title
    t(ax, FW/2, FH - 0.40, "NL Rendering  →  Training JSONL", fs=14, bold=True)
    t(ax, FW/2, FH - 0.85,
      "The same ProofChain produces two training formats depending on the stage.",
      fs=9, c=MID, italic=True)

    # ── ProofChain box ─────────────────────────────────────────────────────────
    PCX, PCY = FW / 2, 6.65
    PCW, PCH = 9.8, 1.00
    rbox(ax, PCX, PCY, PCW, PCH,
         "ProofChain\n"
         "premises: [P→Q,  P,  Q→R]     ·     "
         "steps: [IMPL_ELIM×2]     ·     conclusion: R",
         fs=9.0, lw=1.2)

    # ── Stem down to split ─────────────────────────────────────────────────────
    STEM_BOT = PCY - PCH/2    # = 6.15
    SPLIT_Y  = 5.55
    ax.plot([PCX, PCX], [STEM_BOT, SPLIT_Y], color=BORD, lw=1.0, zorder=5)

    # renderer label (beside the stem)
    t(ax, PCX + 0.18, (STEM_BOT + SPLIT_Y) / 2,
      "NaturalLanguageRenderer.render()", fs=8.0, c=MID,
      italic=True, ha="left")

    # ── Horizontal split arms ─────────────────────────────────────────────────
    LCX = 3.40    # Stage 1 centre-x
    RCX = 10.60   # Stage 2 centre-x

    for cx in [LCX, RCX]:
        ax.plot([PCX, cx], [SPLIT_Y, SPLIT_Y], color=BORD, lw=1.0, zorder=5)
        arr(ax, cx, SPLIT_Y, cx, SPLIT_Y - 0.58, lw=1.0, hw=0.14, hl=0.20)

    # ── Stage column labels ────────────────────────────────────────────────────
    # Labels go just below the arms → y = SPLIT_Y - 0.68
    LBL_Y = SPLIT_Y - 0.72
    t(ax, LCX, LBL_Y, "Stage 1  —  premises + conclusion only",
      fs=10.5, bold=True)
    t(ax, RCX, LBL_Y, "Stage 2  —  full proof trace",
      fs=10.5, bold=True)

    # ── Code block dimensions ─────────────────────────────────────────────────
    CW = 5.8

    # Stage 1: 4 lines of tagged text
    S1_CODE = (
        "<PREMISE> if it rains, the soil is wet </PREMISE>\n"
        "<PREMISE> it rains </PREMISE>\n"
        "<PREMISE> if the soil is wet, plants grow </PREMISE>\n"
        "<CONCLUSION> plants grow </CONCLUSION>"
    )
    S1H = 1.95    # box height for 4-line block
    S1Y = LBL_Y - 0.40 - S1H / 2    # box centre-y

    rbox(ax, LCX, S1Y, CW, S1H, S1_CODE, fs=8.5, mono=True, lw=0.9)

    t(ax, LCX, S1Y - S1H/2 - 0.32,
      "No intermediate steps.  Loss on <CONCLUSION> token(s) only.",
      fs=8.0, c=MID, italic=True)

    # Stage 2: 6 lines of tagged text (includes one intermediate CONCLUSION)
    S2_CODE = (
        "<PREMISE> if it rains, the soil is wet </PREMISE>\n"
        "<PREMISE> it rains </PREMISE>\n"
        "<CONCLUSION> the soil is wet </CONCLUSION>\n"
        "<PREMISE> if the soil is wet, plants grow </PREMISE>\n"
        "<PREMISE> the soil is wet </PREMISE>\n"
        "<CONCLUSION> plants grow </CONCLUSION>"
    )
    S2H = 2.80    # box height for 6-line block
    S2Y = LBL_Y - 0.40 - S2H / 2    # same top-align as Stage 1 ✓

    rbox(ax, RCX, S2Y, CW, S2H, S2_CODE, fs=8.5, mono=True, lw=0.9)

    # ── Annotation: point to the intermediate CONCLUSION line ─────────────────
    # The code has 6 lines; line 3 is the intermediate conclusion.
    # Estimate y: top of box = S2Y + S2H/2; line height ≈ S2H/6
    CODE_TOP = S2Y + S2H / 2
    LINE_H   = S2H / 6.0
    INT_Y    = CODE_TOP - LINE_H * 2.5   # centre of line 3

    ANN_X = RCX + CW / 2 + 0.15
    arr(ax, ANN_X + 1.60, INT_Y, ANN_X + 0.10, INT_Y,
        lw=0.8, hw=0.10, hl=0.14)
    t(ax, ANN_X + 1.68, INT_Y,
      "intermediate conclusion\n"
      "(absent when compression\n"
      " merges the two steps)",
      fs=7.5, c=MID, italic=True, ha="left")

    t(ax, RCX, S2Y - S2H/2 - 0.32,
      "Each inference step explicit.  Loss on all <CONCLUSION> token(s).",
      fs=8.0, c=MID, italic=True)

    out = os.path.join(OUT_DIR, "nl_rendering.png")
    plt.savefig(out, dpi=150, bbox_inches="tight",
                facecolor=BG, pad_inches=0.18)
    plt.close()
    print(f"Saved → {out}")


if __name__ == "__main__":
    fig_compression()
    fig_rendering()
