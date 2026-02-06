"""Rewrite cell 0 of the notebook with the proper monkeypatch code."""

import json

NB_PATH = r"e:\mjosa_complete\gref4hsi\mjosa_code\notebooks\17_compare_dark_vs_empty_section.ipynb"

NEW_CELL_0 = r'''# Monkeypatch: track_start/track_end + clean display + working display_aspect_ratio
import matplotlib.pyplot as plt

def plot_rgb_with_extras(original_plot_rgb):
    """
    Wrapper that adds to plot_rgb:
      - track_start / track_end:     Slice data to a track range
      - display_aspect_ratio:        Stretch pixels to physical aspect (works WITHOUT cropping).
                                     Keeps a fixed figure width, adjusts height.
      - clean=True:                  Remove ALL decoration (axes, labels, title, whitespace,
                                     ticks, spines, gridlines). Just the RGB image.
      - fig_width:                   Figure width in inches (default 15). Height auto-calculated.
    """

    def wrapper(self, track_start=None, track_end=None,
                display_aspect_ratio=None, fig_width=15,
                clean=False, **kwargs):

        # --- 1) Determine data and whether we need to slice ---
        use_corrected = kwargs.get("use_corrected", False)
        if use_corrected and hasattr(self, "data_corrected"):
            current_data = self.data_corrected
        else:
            current_data = self.data
        T_total = current_data.shape[0]

        # --- 2) Slice data if track range specified ---
        need_slice = (track_start is not None or track_end is not None)
        original_data = None
        if need_slice:
            original_data = current_data
            start_idx = track_start if track_start is not None else 0
            end_idx = track_end if track_end is not None else T_total

            if start_idx < 0 or start_idx >= T_total:
                raise ValueError(f"track_start={start_idx} out of range [0, {T_total})")
            if end_idx <= start_idx or end_idx > T_total:
                raise ValueError(f"track_end={end_idx} must be in ({start_idx}, {T_total}]")

            sliced = original_data[start_idx:end_idx, :, :]
            if use_corrected and hasattr(self, "data_corrected"):
                self.data_corrected = sliced
            else:
                self.data = sliced

        # --- 3) Calculate figsize from display_aspect_ratio ---
        # Fix width, compute height so pixels appear with the desired stretch.
        #   display_aspect_ratio = how many times wider a track-pixel appears vs a slit-pixel
        #   For UHI: physical ratio is ~3.61 (track ~5cm, slit ~1.4cm)
        #   height = fig_width * (n_slits / n_tracks) / display_aspect_ratio
        if display_aspect_ratio is not None:
            if use_corrected and hasattr(self, "data_corrected"):
                shape_data = self.data_corrected
            else:
                shape_data = self.data
            n_tracks, n_slits = shape_data.shape[0], shape_data.shape[1]
            fig_height = fig_width * (n_slits / n_tracks) / display_aspect_ratio
            fig_height = max(1.5, min(fig_height, 30))  # clamp to sane range
            kwargs['figsize'] = (fig_width, fig_height)
            print(f"\U0001f4d0 aspect={display_aspect_ratio}: figsize=({fig_width}, {fig_height:.2f}) "
                  f"for {n_tracks}x{n_slits} image")

        # --- 4) If clean mode, force the built-in display params ---
        if clean:
            kwargs['show_ticks'] = False
            kwargs['hide_axes'] = True

        # --- 5) Call original plot_rgb ---
        try:
            result = original_plot_rgb(self, **kwargs)

            # --- 6) Post-hoc cleanup for clean mode ---
            # BUG in original: plt.xlabel/ylabel/title are set AFTER the hide_axes
            # block, so they get re-added even when hide_axes=True. We strip everything.
            if clean:
                fig = plt.gcf()
                ax = plt.gca()
                ax.set_title("")
                ax.set_xlabel("")
                ax.set_ylabel("")
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                ax.grid(False, which="both")
                # Kill ALL whitespace/padding around the image
                fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

            return result
        finally:
            # --- 7) Restore original data if we sliced ---
            if need_slice and original_data is not None:
                if use_corrected and hasattr(self, "data_corrected"):
                    self.data_corrected = original_data
                else:
                    self.data = original_data

    return wrapper


# Apply monkeypatch
CombinedTransectCube.plot_rgb = plot_rgb_with_extras(CombinedTransectCube.plot_rgb)

print("\u2705 Monkeypatch applied. New plot_rgb() parameters:")
print("   track_start / track_end  \u2014 slice to a track range")
print("   display_aspect_ratio     \u2014 pixel stretch (e.g. 3.61), auto-sizes figure")
print("   fig_width                \u2014 figure width in inches (default 15)")
print("   clean=True               \u2014 strip ALL decoration (title, labels, axes, whitespace)")
'''

NEW_CELL_1 = """# Test 1: Clean image - no title, no axes, no labels, no whitespace
cube.plot_rgb(
    use_corrected=True,
    track_start=700,
    track_end=2000,
    display_aspect_ratio=3.61,  # Physical pixel ratio for UHI
    clean=True,                  # Strip ALL decoration
)
"""

NEW_CELL_2 = """# Test 2: Same but with more stretch and wider figure
cube.plot_rgb(
    use_corrected=True,
    track_start=700,
    track_end=2000,
    display_aspect_ratio=5.0,   # More stretched
    fig_width=20,                # Wider figure
    clean=True,
)
"""

with open(NB_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)


# Replace cell 0 (monkeypatch), cell 1 (test 1), cell 2 (test 2)
def to_source(code):
    """Convert a code string to notebook source format (list of lines with \n)."""
    lines = code.split("\n")
    result = []
    for i, line in enumerate(lines):
        if i < len(lines) - 1:
            result.append(line + "\n")
        elif line:  # Skip empty trailing line
            result.append(line)
    return result


nb["cells"][0]["source"] = to_source(NEW_CELL_0)
nb["cells"][1]["source"] = to_source(NEW_CELL_1)
nb["cells"][2]["source"] = to_source(NEW_CELL_2)

# Clear outputs on modified cells
for i in range(3):
    nb["cells"][i]["outputs"] = []
    nb["cells"][i]["execution_count"] = None

with open(NB_PATH, "w", encoding="utf-8", newline="\n") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("✅ Notebook cells 0, 1, 2 rewritten successfully")
print(f"   Cell 0: {len(nb['cells'][0]['source'])} lines (monkeypatch)")
print(f"   Cell 1: {len(nb['cells'][1]['source'])} lines (test 1)")
print(f"   Cell 2: {len(nb['cells'][2]['source'])} lines (test 2)")
