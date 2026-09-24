import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from src.data.loader import *
from src.data.grid import RamanGrid
from src.cnn import runs
from src import paths
from src.visualisation.view_heatmap import show_hsi_viewer
from src.visualisation.view_phase_decomp import show_spectra_ext_viewer
from src.visualisation.view_predict import show_prediction_map

HELP_TEXT = {
    "args": """
  PATH: Absolute or relative path to your data directory.
        Accepts ~ for home directory (e.g. ~/data/SB008).
        The directory should contain .txt spectral data files.
        Example: ~/Data/SB008  or  /home/user/project/data/SB008

  ROWS (x): Number of rows in the grid (integer).
  COLS (y): Number of columns in the grid (integer).

  Example input: ~/Data/SB008 10 13
""",
    "mode": """
  Modes:
    1 - Heatmap:   Generate a spatial heatmap from spectral data.
    2 - Unmixing:  Decompose spectra into end-member components.
    3 - Predict:   Run prediction pipeline (incomplete).
""",
    "heatmap": """
  PIPELINE: Pre-processing pipeline to apply before plotting.

    0 - None (raw data)

    1 - P1: First preprocessing protocol from Georgiev et al. (2023) [1].
        Steps:
          - Cosmic ray removal (Whitaker-Hayes algorithm)
          - Denoising (Gaussian filter)
          - Baseline correction (Asymmetric Least Squares)
          - Normalisation (Area under the curve)

    2 - P2: Second preprocessing protocol from Georgiev et al. (2023) [1].
        Steps:
          - Cosmic ray removal (Whitaker-Hayes algorithm)
          - Denoising (Savitzky-Golay filter, window=9, poly order=3)
          - Baseline correction (Adaptive Smoothness Penalized Least Squares)
          - Normalisation (MinMax)

    3 - P3: Third preprocessing protocol from Georgiev et al. (2023) [1].
        Steps:
          - Cosmic ray removal (Whitaker-Hayes algorithm)
          - Baseline correction (polynomial fitting, order=3)
          - Normalisation (Vector)

  References:
    [1] Georgiev et al. (2023). RamanSPy: An open-source Python package for integrative
        Raman spectroscopy data analysis. arXiv:2307.13650.

""",
    "phase_decomp": """
  END MEMBERS: Number of spectral components to unmix into.
               Enter -1 to estimate automatically via SVD.
  START / END: Wavenumber range to analyse in cm⁻¹. Press Enter to use full range.
""",
    "predict": """
  CONFIDENCE THRESHOLD: Minimum confidence (0-1) for a prediction to be displayed.
                        Pixels below this threshold are shown as 'Unknown'.
                        Example: 0.80 means only predictions ≥80% confidence are shown.
  SAVE PATH: File path to save the prediction map image.
             Press Enter to use the default, which is prediction_map.png in the
             outputs directory next to the repository, wherever you run this from.
""",
    "run": """
  RUN: The directory holding a trained model, written by training. It holds the
       model, the label encoder and a run.json recording the input length and the
       wavenumber range the model was trained on, which is where this mode reads
       those settings from. Pressing Enter uses the most recently created run.
""",
}

def _input(prompt_str=""):
    """Thin wrapper so callers can check for '?' and show help externally."""
    return input(prompt_str).strip()

def prompt_args():
    print("\nEnter path, grid rows and grid columns (or '?' for help):")
    print("e.g. ~/Code/Data_SH/SB008 10 13")
    while True:
        raw = _input("> ")
        if raw == "?":
            print(HELP_TEXT["args"])
            continue
        parts = raw.split()
        if len(parts) != 3:
            print("Please enter a path and two integers (rows, cols), or '?' for help.")
            continue
        path, x, y = parts
        try:
            x, y = int(x), int(y)
        except ValueError:
            print("x and y must be integers, please try again.")
            continue

        path = os.path.expanduser(path)

        if not os.path.isdir(path):
            print(f"Directory not found: {path}")
            print("Please check the path and try again.")
            continue

        # The grid owns file discovery, so the CLI counts exactly what the loaders
        # will read: numbered files only, and exactly rows x cols of them.
        try:
            files = RamanGrid(path, x, y).get_sorted_files()
        except (FileNotFoundError, ValueError) as exc:
            print(exc)
            print("Please check the path and the dimensions and try again.")
            continue

        print(f"Found {len(files)} numbered .txt file(s) in {path}.")
        break

    mode = prompt_mode()
    if mode == "heatmap":
        kwargs = prompt_heatmap_args()
    elif mode == "phase_decomp":
        kwargs = prompt_phase_decomp_args()
    elif mode == "predict":
        kwargs = prompt_predict_args()

    return path, x, y, mode, kwargs

def prompt_mode():
    print("\nSelect mode (or '?' for help):")
    print("  [1] Heatmap")
    print("  [2] Unmixing")
    print("  [3] Predict (incomplete)")
    choices = {"1": "heatmap", "2": "phase_decomp", "3": "predict"}
    while True:
        val = _input("> ")
        if val == "?":
            print(HELP_TEXT["mode"])
            continue
        if val in choices:
            return choices[val]
        print("Invalid choice, please enter 1, 2, or 3.")

def prompt_heatmap_args():
    print("\nPipeline? [0] None  [1] P1  [2] P2  [3] P3 (default: 0, or '?' for help)")
    while True:
        pipeline_id = _input("> ") or "0"
        if pipeline_id == "?":
            print(HELP_TEXT["heatmap"])
            continue
        if pipeline_id in ("0", "1", "2", "3"):
            break
        print("Please enter 0, 1, 2, or 3.")

    print("\nStart of spectra in cm⁻¹ (press Enter to skip, or '?' for help)")
    while True:
        start = _input("> ") or None
        if start == "?":
            print(HELP_TEXT["heatmap"])
            continue
        try:
            start = float(start) if start else None
            break
        except ValueError:
            print("Please enter a number or press Enter to skip.")

    print("\nEnd of spectra in cm⁻¹ (press Enter to skip, or '?' for help)")
    while True:
        end = _input("> ") or None
        if end == "?":
            print(HELP_TEXT["heatmap"])
            continue
        try:
            end = float(end) if end else None
            break
        except ValueError:
            print("Please enter a number or press Enter to skip.")

    return {
        "pipeline_id": int(pipeline_id),
        "start": start,
        "end": end,
    }

def prompt_phase_decomp_args():
    print("\nNumber of phases (-1 to estimate automatically, or '?' for help)")
    while True:
        phase_number = _input("> ") or "-1"
        if phase_number == "?":
            print(HELP_TEXT["phase_decomp"])
            continue
        try:
            phase_number = int(phase_number)
            break
        except ValueError:
            print("Please enter an integer, or -1 to estimate automatically.")

    print("\nStart of spectra in cm⁻¹ (press Enter to skip, or '?' for help)")
    while True:
        start = _input("> ") or None
        if start == "?":
            print(HELP_TEXT["phase_decomp"])
            continue
        try:
            start = float(start) if start else None
            break
        except ValueError:
            print("Please enter a number or press Enter to skip.")

    print("\nEnd of spectra in cm⁻¹ (press Enter to skip, or '?' for help)")
    while True:
        end = _input("> ") or None
        if end == "?":
            print(HELP_TEXT["phase_decomp"])
            continue
        try:
            end = float(end) if end else None
            break
        except ValueError:
            print("Please enter a number or press Enter to skip.")

    return {
        "phase_number": phase_number,
        "start": start,
        "end": end,
    }

def prompt_run_dir():
    """Which trained run to predict with. Returns a path to a run directory."""
    names = runs.list_runs()

    if not names:
        print("\nNo trained runs found. Training writes one directory per model, and")
        print("this mode reads the model's settings from the run.json inside it.")
        while True:
            raw = _input("Run directory: ")
            if raw == "?":
                print(HELP_TEXT["run"])
                continue
            if raw:
                return raw
            print("Please enter the path to a run directory, or '?' for help.")

    print("\nWhich trained run? (Enter for the most recent, or '?' for help)")
    for name in names:
        print(f"  {name}")
    while True:
        raw = _input("> ") or names[0]
        if raw == "?":
            print(HELP_TEXT["run"])
            continue
        if raw in names:
            return str(runs.run_directory(raw))
        if os.path.isdir(os.path.expanduser(raw)):
            return os.path.expanduser(raw)
        print("Please enter one of the names above, or the path to a run directory.")


def prompt_predict_args():
    run_path = prompt_run_dir()

    print("\nConfidence threshold (default: 0.80, or '?' for help)")
    while True:
        confidence_threshold = _input("> ") or "0.80"
        if confidence_threshold == "?":
            print(HELP_TEXT["predict"])
            continue
        try:
            confidence_threshold = float(confidence_threshold)
            if 0 < confidence_threshold <= 1:
                break
            print("Please enter a value between 0 and 1 (e.g. 0.80).")
        except ValueError:
            print("Please enter a number between 0 and 1 (e.g. 0.80).")

    default_save_path = str(paths.output_path("prediction_map.png"))
    print(f"\nSave path for prediction map (default: {default_save_path}, or '?' for help)")
    while True:
        save_path = _input("> ") or default_save_path
        if save_path == "?":
            print(HELP_TEXT["predict"])
            continue
        break

    return {
        "run_path": run_path,
        "confidence_threshold": confidence_threshold,
        "save_path": save_path,
    }

def main():
    path, x, y, mode, kwargs = prompt_args()

    if mode == "heatmap":
        cumulative_cube, spectra_of_pixel, raman_shift, pixel_map = get_cumulative_hsi_cube(
            path, x, y, kwargs["pipeline_id"],
            kwargs["start"], kwargs["end"]
        )
        show_hsi_viewer(cumulative_cube, spectra_of_pixel, raman_shift, pixel_map, x, y)

    elif mode == "phase_decomp":
        hsi_cube = get_raw_hsi_cube(path, x, y)
        show_spectra_ext_viewer(hsi_cube, kwargs["phase_number"], kwargs["start"], kwargs["end"])

    elif mode == "predict":
        show_prediction_map(
            kwargs["run_path"], path, x, y,
            confidence_threshold=kwargs["confidence_threshold"],
            save_path=kwargs["save_path"],
        )
        
if __name__ == "__main__":
    main()