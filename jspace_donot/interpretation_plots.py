"""Publication panels retained from the supplied interpretation notebook."""
import numpy as np

def export_original_panels(coefficients, mean_absolute_ig, impact, output):
    coef=np.asarray(coefficients).reshape(4,4)
    mean_abs=np.asarray(mean_absolute_ig).reshape(4,4)
    OUTPUT=output
    # import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    from matplotlib.colors import LinearSegmentedColormap
    from pathlib import Path

    # ============================================================
    # JSpaceDoNOT interpretability figure
    # ChatGPT-interface-like compact tile design
    # No subplot titles; caption should be added in LaTeX
    # ============================================================

    OUTPUT = Path(OUTPUT)
    OUTPUT.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # Input data
    # -----------------------------
    LR_VALUES = np.asarray(coef, dtype=float).reshape(4, 4)
    IG_VALUES = np.asarray(mean_abs, dtype=float).reshape(4, 4)
    PERT_VALUES = (
        impact["mlp_mean_abs_delta_p"]
        .to_numpy(dtype=float)
        .reshape(4, 4) * 100.0   # percentage points
    )

    # -----------------------------
    # Figure style parameters
    # -----------------------------
    FIG_BG = "#FFFFFF"
    TEXT_DARK = "#2F3A4A"
    TEXT_LIGHT = "#6E7B8C"
    TEXT_FAINT = "#8F99A8"

    # These are chosen to visually resemble the soft UI tone:
    # - negative: cool slate/blue
    # - neutral: near-white
    # - positive: muted salmon/red
    NEG_CMAP = LinearSegmentedColormap.from_list(
        "neg_soft",
        ["#F6F8FC", "#DCE5F6", "#AFC2EA", "#6E8ED7", "#4364B4"]
    )
    POS_CMAP = LinearSegmentedColormap.from_list(
        "pos_soft",
        ["#FDF8F6", "#F5DDD8", "#E9B8AE", "#D77E6E", "#C65A4A"]
    )
    SEQ_CMAP = LinearSegmentedColormap.from_list(
        "seq_soft",
        ["#F3FAF8", "#D9EFE7", "#A9D9C9", "#66B59A", "#2B8D73", "#166454"]
    )

    # Choose layout:
    # "wide" for 3 panels in one row
    # "column" for 3 stacked panels
    LAYOUT = "wide"

    if LAYOUT == "wide":
        fig, axes = plt.subplots(
            1, 3,
            figsize=(6.9, 1.95),
            gridspec_kw={"wspace": 0.08}
        )
    else:
        fig, axes = plt.subplots(
            3, 1,
            figsize=(2.55, 6.2),
            gridspec_kw={"hspace": 0.16}
        )

    fig.patch.set_facecolor(FIG_BG)

    panels = [
        ("lr", LR_VALUES),
        ("ig", IG_VALUES),
        ("pert", PERT_VALUES),
    ]

    def draw_tile_heatmap(ax, data, mode, annotate_fmt):
        ax.set_xlim(-0.22, 4.02)
        ax.set_ylim(4.02, -0.22)
        ax.set_aspect("equal")
        ax.axis("off")

        # Column labels
        for j in range(4):
            ax.text(
                j + 0.5, -0.11, f"R{j+1}",
                ha="center", va="center",
                fontsize=7.9,
                color=TEXT_LIGHT
            )

        # Row labels
        for i in range(4):
            ax.text(
                -0.10, i + 0.5, f"L{i+1}",
                ha="right", va="center",
                fontsize=7.9,
                color=TEXT_LIGHT
            )

        if mode == "signed":
            vmax = max(float(np.abs(data).max()), 1e-8)
        else:
            vmax = max(float(data.max()), 1e-8)

        for i in range(4):
            for j in range(4):
                v = float(data[i, j])

                if mode == "signed":
                    if v >= 0:
                        strength = abs(v) / vmax
                        face = POS_CMAP(0.10 + 0.86 * strength)
                    else:
                        strength = abs(v) / vmax
                        face = NEG_CMAP(0.10 + 0.86 * strength)
                else:
                    strength = v / vmax
                    face = SEQ_CMAP(0.08 + 0.88 * strength)

                # Rounded tiles with small gaps
                tile = FancyBboxPatch(
                    (j + 0.07, i + 0.07),
                    0.86, 0.86,
                    boxstyle="round,pad=0.015,rounding_size=0.12",
                    linewidth=0,
                    facecolor=face,
                    edgecolor=None,
                    antialiased=True
                )
                ax.add_patch(tile)

                # Text color based on tile darkness
                txt_color = "#FFFFFF" if strength > 0.58 else TEXT_DARK

                ax.text(
                    j + 0.5, i + 0.5,
                    annotate_fmt(v),
                    ha="center", va="center",
                    fontsize=8.0,
                    color=txt_color,
                    fontweight="medium"
                )

    for ax, (kind, arr) in zip(axes, panels):
        if kind == "lr":
            draw_tile_heatmap(
                ax, arr, mode="signed",
                annotate_fmt=lambda x: f"{x:+.2f}"
            )
        elif kind == "ig":
            draw_tile_heatmap(
                ax, arr, mode="positive",
                annotate_fmt=lambda x: f"{x:.2f}"
            )
        else:
            draw_tile_heatmap(
                ax, arr, mode="positive",
                annotate_fmt=lambda x: f"{x:.1f}"
            )

    # Very tight outer margins
    if LAYOUT == "wide":
        fig.subplots_adjust(
            left=0.018,
            right=0.998,
            top=0.985,
            bottom=0.055,
            wspace=0.05
        )
    else:
        fig.subplots_adjust(
            left=0.06,
            right=0.995,
            top=0.995,
            bottom=0.025,
            hspace=0.08
        )

    base = OUTPUT / f"jspace_interpretability_chatgpt_style_{LAYOUT}"

    # ============================================================
    # Export three independent, tightly cropped PDF panels
    # Requires the existing draw_tile_heatmap() and palettes
    # ============================================================

    export_panels = [
        ("jspace_lr_coefficients", LR_VALUES, "signed", lambda x: f"{x:+.2f}"),
        ("jspace_mlp_ig", IG_VALUES, "positive", lambda x: f"{x:.2f}"),
        ("jspace_mlp_intervention", PERT_VALUES, "positive", lambda x: f"{x:.1f}"),
    ]

    for name, data, mode, formatter in export_panels:
        fig, ax = plt.subplots(figsize=(2.5, 2.5))
        fig.patch.set_facecolor("white")

        # Reuse exactly the same rendering style
        draw_tile_heatmap(ax, data, mode, formatter)

        # Tight layout, without titles or subtitles
        fig.subplots_adjust(
            left=0.13,
            right=0.995,
            bottom=0.01,
            top=0.995
        )

        for extension in ("pdf", "svg", "png"):
            fig.savefig(
                OUTPUT / f"{name}.{extension}",
                dpi=400 if extension == "png" else None,
                bbox_inches="tight",
                pad_inches=0.015,
                facecolor="white"
            )

        plt.close(fig)
        plt.close(fig)

    print("Saved all three independent heatmaps.")

    # ============================================================
    # Appendix: LR vs MLP Feature Sensitivity
    # Same soft ChatGPT-style tiles as the main figure
    # Two independent PDFs, no titles
    # ============================================================

    OUTPUT = Path(OUTPUT)
    OUTPUT.mkdir(parents=True, exist_ok=True)

    # Read directly from the original sensitivity experiment
    assert len(impact) == 16

    ordered = impact.sort_values(["layer", "rank"])

    LR_SENS = (
        ordered["lr_mean_abs_delta_p"]
        .to_numpy(dtype=float)
        .reshape(4, 4) * 100
    )

    MLP_SENS = (
        ordered["mlp_mean_abs_delta_p"]
        .to_numpy(dtype=float)
        .reshape(4, 4) * 100
    )

    # Both use the same color scale, so values are comparable
    COMMON_MAX = max(
        LR_SENS.max(),
        MLP_SENS.max(),
        1e-8
    )

    TEAL = LinearSegmentedColormap.from_list(
        "jspace_soft_teal",
        [
            "#F3FAF8",
            "#D9EFE7",
            "#A9D9C9",
            "#66B59A",
            "#2B8D73",
            "#166454"
        ]
    )

    TEXT_DARK = "#2F3A4A"
    TEXT_LIGHT = "#6E7B8C"

    def plot_sensitivity_tiles(data, output_name):

        fig, ax = plt.subplots(figsize=(2.5, 2.5))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        ax.set_xlim(-0.22, 4.02)
        ax.set_ylim(4.02, -0.22)
        ax.set_aspect("equal")
        ax.axis("off")

        # Column labels
        for j in range(4):
            ax.text(
                j + 0.5, -0.11,
                f"R{j+1}",
                ha="center",
                va="center",
                fontsize=8,
                color=TEXT_LIGHT
            )

        # Row labels
        for i in range(4):
            ax.text(
                -0.10, i + 0.5,
                f"L{i+1}",
                ha="right",
                va="center",
                fontsize=8,
                color=TEXT_LIGHT
            )

        for i in range(4):
            for j in range(4):

                value = float(data[i, j])
                strength = np.clip(value / COMMON_MAX, 0, 1)

                color = TEAL(0.08 + 0.88 * strength)

                tile = FancyBboxPatch(
                    (j + 0.07, i + 0.07),
                    0.86,
                    0.86,
                    boxstyle=(
                        "round,pad=0.015,"
                        "rounding_size=0.12"
                    ),
                    linewidth=0,
                    facecolor=color,
                    antialiased=True
                )

                ax.add_patch(tile)

                text_color = (
                    "#FFFFFF"
                    if strength > 0.58
                    else TEXT_DARK
                )

                ax.text(
                    j + 0.5,
                    i + 0.5,
                    f"{value:.1f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    fontweight="medium",
                    color=text_color
                )

        fig.subplots_adjust(
            left=0.018,
            right=0.998,
            top=0.995,
            bottom=0.015
        )

        for ext in ["pdf", "svg", "png"]:
            fig.savefig(
                OUTPUT / f"{output_name}.{ext}",
                dpi=400 if ext == "png" else None,
                bbox_inches="tight",
                pad_inches=0.015,
                facecolor="white"
            )

        plt.close(fig)
        plt.close(fig)


    plot_sensitivity_tiles(
        LR_SENS,
        "appendix_lr_feature_sensitivity"
    )

    plot_sensitivity_tiles(
        MLP_SENS,
        "appendix_mlp_feature_sensitivity"
    )

    print("Saved Appendix figures to:", OUTPUT)
    print("Shared color scale maximum:", COMMON_MAX)
    plt.close("all")


def plot_score_sweep(sweep,feature_index,output,cutoff=.85):
    """Actual input log-score versus STOP probability, including threshold crossings."""
    import matplotlib.pyplot as plt
    from pathlib import Path
    x=sweep[f'feature_{feature_index}_raw_score'].to_numpy();y=sweep.probability.to_numpy()
    order=np.argsort(x);x=x[order];y=y[order]
    crossings=[]
    for i in range(len(x)-1):
        a=y[i]-cutoff;b=y[i+1]-cutoff
        if a==0:crossings.append(float(x[i]))
        elif a*b<0:crossings.append(float(x[i]+(x[i+1]-x[i])*(-a)/(b-a)))
    if len(x) and y[-1]==cutoff:crossings.append(float(x[-1]))
    fig,ax=plt.subplots(figsize=(4.3,3))
    ax.plot(x,y,color='#168577',lw=1.8)
    ax.axhline(cutoff,color='#A4ADB5',ls='--',lw=1)
    if crossings:ax.scatter(crossings,[cutoff]*len(crossings),color='#C77743',zorder=3)
    original=sweep[np.isclose(sweep.scale,1.)]
    if not original.empty:ax.scatter(original[f'feature_{feature_index}_raw_score'],original.probability,color='#4477AA',zorder=4)
    ax.set(xlabel='Actual J++ input log-score',ylabel='STOP probability',ylim=(-.03,1.03))
    ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.15);fig.tight_layout()
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    for ext in ['png','pdf']:fig.savefig(output.with_suffix('.'+ext),dpi=300,bbox_inches='tight')
    plt.close(fig)
    return crossings
