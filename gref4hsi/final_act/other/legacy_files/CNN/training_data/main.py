from helper import *

if __name__ == "__main__":

    # --- CONFIGURATION ---
    data_folder = r"E:\mjosa\29\uhi_rad\active_training_data\not_annotated"

    # annotate only sediment vs dark, rest is skipped
    radiance_out = r"E:\mjosa\29\training_data_new\fundation\radiance"
    reflectance_out = r"E:\mjosa\29\training_data_new\fundation\reflectance"

    # ==============================  this is for next time  ==============================
    # # annotate bomb
    # radiance_out = r"E:\mjosa\29\training_data_new\bomb\radiance"
    # reflectance_out = r"E:\mjosa\29\training_data_new\bomb\reflectance"

    # # annotate fish
    # radiance_out = r"E:\mjosa\29\training_data_new\fish\radiance"
    # reflectance_out = r"E:\mjosa\29\training_data_new\fish\reflectance"

    # =====================================================================================

    roi_plate_file = r"E:\mjosa\29\ROIs\statistics_rad\reflectance_plate.txt"

    segment_length = 50

    files = sorted(f for f in os.listdir(data_folder) if f.endswith(".h5"))
    print(f"\n🔍 Found {len(files)} transects in {data_folder}")
    print(
        "   Categories:",
        "bomb,",
        "dark,",
        "sediment,",
        "fish,",
        "other,",
        "| back = previous, skip = no label",
    )

    for fn in files:
        name = os.path.splitext(fn)[0]
        path = os.path.join(data_folder, fn)
        print(f"\n============================")
        print(f"📦 Processing transect '{name}'")
        print(f"============================\n")

        # 1) Load just this cube
        cube = UHIFile(path)

        # 2) Radiance pass (interactive)
        print("🔶 Radiance labeling …")
        radi_dir = os.path.join(radiance_out, name)
        os.makedirs(radi_dir, exist_ok=True)
        cube.create_training_data(
            out_dir=radi_dir,
            segment_length=segment_length,
            reuse_labels=False,
            labels_file=None,
        )

        # 3) Reflectance pass (chunked per-segment)
        labels_csv = os.path.join(radiance_out, name, "labels.csv")
        try:
            df_labels = pd.read_csv(labels_csv)
        except pd.errors.EmptyDataError:
            print(
                f"[⚠️ WARNING] No labels in {labels_csv}, skipping reflectance for '{name}'"
            )
            continue

        print("\n🔷 Reflectance labeling (chunked)…")
        refl_dir = os.path.join(reflectance_out, name)
        os.makedirs(refl_dir, exist_ok=True)

        for rec in df_labels.itertuples():
            start, end, label = rec.start, rec.end, rec.label

            # slice only this small block
            rel_start = start - cube.track_offset
            rel_end = end - cube.track_offset
            seg_cube = cube.slice_tracks(rel_start, rel_end)

            # convert to reflectance
            refl_seg = seg_cube.get_reflectance(roi_plate_file)

            # remove old
            pattern = f"_{start}_{end}.npz"
            for old in os.listdir(refl_dir):
                if old.endswith(pattern):
                    os.remove(os.path.join(refl_dir, old))

            # save new (float32 to cut memory in half)
            fname = f"{label}_{start}_{end}.npz"
            out_path = os.path.join(refl_dir, fname)
            np.savez(
                out_path,
                data=refl_seg.data.astype(np.float32),
                wavelengths=refl_seg.wavelengths,
                track_offset=refl_seg.track_offset,
            )
            print(f"   ✅ Saved {fname}")

            # free memory
            del seg_cube, refl_seg
            gc.collect()

        print("✅ Reflectance pass done.")

        # 4) Free the big cube & figures
        import matplotlib.pyplot as plt

        plt.close("all")
        del cube
        gc.collect()

    print("\n✅ All transects processed!")
