"""
Test script to visualize color consistency between 057 and 028 classification schemes.

This script creates a visual comparison of the colors used in:
1. 057 (3-class model): dark, sediment, bombs
2. 028 (6-class model): sediment, rust, dark_bomb, dark_pit, halo, uncertain

Purpose: Ensure consistent colors across both models for paper figures.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Colors from georef.py HARDCODED_ROI_COLORS (hex)
georef_colors_hex = {
    "training_bombs": "#1E90FF",  # dodger blue
    "training_dark": "#000000",  # black
    "training_sediment": "#8B4513",  # brown
    "classified_bombs": "#1E90FF",  # dodger blue
    "classified_dark": "#000000",  # black
    "classified_sediment": "#8B4513",  # brown
    "filtered_bombs": "#1E90FF",  # dodger blue
}

# Colors from notebook 9 plot_classification_as_image (RGB 0-1)
notebook_colors_rgb = {
    "sediment": [0.6, 0.4, 0.2],  # Brown (same as 057)
    "rust": [0.8, 0.2, 0.1],  # Red-orange (rust halo)
    "dark_bomb": [0.1, 0.1, 0.1],  # Super black
    "dark_pit": [0.25, 0.25, 0.25],  # Slightly lighter gray
    "halo": [0.9, 0.9, 0.5],  # Yellow
    "uncertain": [0.5, 0.0, 0.5],  # Purple
    "dark": [0.1, 0.1, 0.1],  # Same as dark_bomb
    "bombs": [0.12, 0.56, 0.93],  # Dodger blue
}


def hex_to_rgb(hex_color):
    """Convert hex color to RGB 0-1 format."""
    hex_color = hex_color.lstrip("#")
    return [int(hex_color[i : i + 2], 16) / 255.0 for i in (0, 2, 4)]


def plot_color_comparison():
    """Create a visual comparison of all colors used in classification."""

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10))

    # Panel 1: 057 georef.py colors (training/classified/filtered)
    ax1.set_title(
        "057 Classification Colors (georef.py HARDCODED_ROI_COLORS)", fontweight="bold"
    )
    y_pos = 0
    for name, hex_color in georef_colors_hex.items():
        rgb = hex_to_rgb(hex_color)
        rect = mpatches.Rectangle(
            (0, y_pos), 1, 1, facecolor=rgb, edgecolor="black", linewidth=2
        )
        ax1.add_patch(rect)
        ax1.text(
            1.1,
            y_pos + 0.5,
            f"{name}: {hex_color} → RGB{tuple(np.round(rgb, 2))}",
            va="center",
            fontsize=10,
        )
        y_pos += 1.2

    ax1.set_xlim(0, 5)
    ax1.set_ylim(0, y_pos)
    ax1.axis("off")

    # Panel 2: 028 notebook colors
    ax2.set_title(
        "028 Classification Colors (notebook 9 plot_classification_as_image)",
        fontweight="bold",
    )
    y_pos = 0
    for name, rgb in notebook_colors_rgb.items():
        rect = mpatches.Rectangle(
            (0, y_pos), 1, 1, facecolor=rgb, edgecolor="black", linewidth=2
        )
        ax2.add_patch(rect)
        # Convert RGB back to hex for display
        hex_color = "#{:02x}{:02x}{:02x}".format(
            int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
        )
        ax2.text(
            1.1,
            y_pos + 0.5,
            f"{name}: RGB{tuple(rgb)} → {hex_color}",
            va="center",
            fontsize=10,
        )
        y_pos += 1.2

    ax2.set_xlim(0, 5)
    ax2.set_ylim(0, y_pos)
    ax2.axis("off")

    # Panel 3: Side-by-side comparison for common classes
    ax3.set_title("Color Consistency Check (Common Classes)", fontweight="bold")

    common_classes = {
        "dark": ("training_dark", "dark"),
        "sediment": ("training_sediment", "sediment"),
        "bombs": ("training_bombs", "bombs"),
    }

    y_pos = 0
    for label, (georef_name, notebook_name) in common_classes.items():
        # Georef color
        hex_color = georef_colors_hex.get(georef_name, "#000000")
        rgb_georef = hex_to_rgb(hex_color)
        rect1 = mpatches.Rectangle(
            (0, y_pos), 1, 1, facecolor=rgb_georef, edgecolor="black", linewidth=2
        )
        ax3.add_patch(rect1)

        # Notebook color
        rgb_notebook = notebook_colors_rgb.get(notebook_name, [0, 0, 0])
        rect2 = mpatches.Rectangle(
            (1.2, y_pos), 1, 1, facecolor=rgb_notebook, edgecolor="black", linewidth=2
        )
        ax3.add_patch(rect2)

        # Check if colors match (within tolerance)
        match = np.allclose(rgb_georef, rgb_notebook, atol=0.02)
        match_text = "✅ MATCH" if match else "❌ MISMATCH"

        ax3.text(
            2.4,
            y_pos + 0.5,
            f"{label.upper()}: {match_text}",
            va="center",
            fontsize=12,
            fontweight="bold",
        )
        ax3.text(0.5, y_pos - 0.15, "057\ngeoref", ha="center", fontsize=8, va="top")
        ax3.text(1.7, y_pos - 0.15, "028\nnotebook", ha="center", fontsize=8, va="top")

        y_pos += 1.5

    ax3.set_xlim(0, 5)
    ax3.set_ylim(-0.5, y_pos)
    ax3.axis("off")

    plt.tight_layout()
    plt.savefig("test_color_consistency_output.png", dpi=150, bbox_inches="tight")
    print("✅ Color comparison plot saved to: test_color_consistency_output.png")
    plt.show()


if __name__ == "__main__":
    print("=" * 80)
    print("COLOR CONSISTENCY TEST")
    print("=" * 80)
    print(
        "\n📊 Checking color consistency between 057 and 028 classification schemes..."
    )
    print("\n🎨 Colors updated:")
    print("   • 057 bombs: RED → BLUE (#1E90FF)")
    print("   • 057 dark: BLACK (unchanged)")
    print("   • 057 sediment: BROWN (unchanged)")
    print("   • 028 dark_pit: Slightly lighter gray")
    print("   • 028 uncertain: GRAY → PURPLE")
    print("   • 028 bombs: BLUE (added for consistency)")
    print("\n" + "=" * 80)

    plot_color_comparison()
