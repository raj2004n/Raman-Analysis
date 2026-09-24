import numpy as np
import matplotlib.pyplot as plt
from src.cnn.predict import predict
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap
from .theme import apply_theme, BG, FG, GRID, ACCENT

def _hover_annotation(row, col, masked_labels, confidence_map, predicted_top5_map,
                      confidence_threshold):
    """What to show for the pixel under the pointer.

    Args:
        row (int): Row index, already rounded from the pointer position.
        col (int): Column index, already rounded from the pointer position.
        masked_labels (np.ndarray): Predicted mineral per pixel, 'Unknown' where the
            prediction fell below the confidence threshold.
        confidence_map (np.ndarray): Top-1 probability per pixel.
        predicted_top5_map (np.ndarray): The five most likely minerals per pixel.
        confidence_threshold (float): Below this, a pixel reads as 'Unknown'.

    Returns:
        tuple[list[str], str] | None: The lines to display and the mineral the pixel
            resolved to, or None when ``(row, col)`` is not a pixel of the map.

    Rounding the pointer position lands exactly on ``width`` or ``length`` when it sits
    on the far edge, so the bounds test has to come before any indexing. Kept apart
    from the matplotlib callback so that rule can be tested without a figure window.
    """
    length, width = masked_labels.shape
    if not (0 <= row < length and 0 <= col < width):
        return None

    confidence = confidence_map[row, col]
    if confidence < confidence_threshold:
        lines = [
            f"Pixel ({row}, {col})",
            "",
            "Unknown",
            f"Confidence: {confidence*100:.1f}% (below threshold)",
        ]
    else:
        lines = [f"Pixel ({row}, {col})", ""]
        for rank, (label, prob) in enumerate(predicted_top5_map[row, col][:5], 1):
            lines.append(f"{rank}. {label}: {prob*100:.1f}%")
    return lines, masked_labels[row, col]


def show_prediction_map(run_path, path, x, y, confidence_threshold=0.80, save_path=None):
    apply_theme()

    predicted_labels_map, predicted_top5_map = predict(run_path, path, x, y)
    
    length, width = predicted_labels_map.shape

    # build confidence map from top-1 probability
    confidence_map = np.array([
        [predicted_top5_map[r, c][0][1] for c in range(width)]
        for r in range(length)
    ])

    # mask low confidence pixels as 'Unknown'
    masked_labels = predicted_labels_map.copy()
    masked_labels[confidence_map < confidence_threshold] = 'Unknown'

    # map mineral names to integers for colouring
    unique_minerals = sorted([m for m in np.unique(masked_labels) if m != 'Unknown'])
    if 'Unknown' in np.unique(masked_labels):
        unique_minerals = unique_minerals + ['Unknown']

    mineral_to_int = {m: i for i, m in enumerate(unique_minerals)}
    label_map = np.vectorize(mineral_to_int.get)(masked_labels)

    n_minerals = len(unique_minerals)
    colors = [plt.get_cmap('tab20')(i / max(n_minerals - 1, 1)) for i in range(n_minerals)]
    if 'Unknown' in unique_minerals:
        colors[-1] = (0.15, 0.15 , 0.18, 1)

    cmap = ListedColormap(colors)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(label_map, cmap=cmap, vmin=0, vmax=n_minerals - 1)
    ax.set_axis_off()
    ax.set_title(f"Mineral Prediction Map (confidence threshold: {confidence_threshold*100:.0f}%)")

    # legend
    legend_elements = [
        Patch(facecolor=colors[i], label=mineral)
        for i, mineral in enumerate(unique_minerals)
    ]
    ax.legend(
        handles=legend_elements,
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        fontsize=8,
        title="Minerals",
        title_fontsize=9,
        framealpha=0.9
    )

    # hover annotation
    annot = ax.annotate(
        "", xy=(0, 0), xytext=(10, 10),
        textcoords="offset points",
        bbox=dict(boxstyle="round", fc="white", alpha=0.9),
        fontsize=8,
        color=FG
    )
    annot.set_visible(False)

    def on_hover(event):
        if event.inaxes == ax and event.xdata is not None and event.ydata is not None:
            row = int(round(event.ydata))
            col = int(round(event.xdata))
            annotation = _hover_annotation(
                row, col, masked_labels, confidence_map, predicted_top5_map,
                confidence_threshold,
            )
            if annotation is None:
                annot.set_visible(False)
            else:
                lines, mineral = annotation
                annot.xy = (event.xdata, event.ydata)
                annot.set_text("\n".join(lines))
                annot.get_bbox_patch().set_facecolor(colors[mineral_to_int[mineral]])
                annot.get_bbox_patch().set_edgecolor(FG)
                annot.set_visible(True)
        else:
            annot.set_visible(False)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", on_hover)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()