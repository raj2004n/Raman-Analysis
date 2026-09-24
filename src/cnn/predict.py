import numpy as np
from src.cnn import runs
from src.data.loader import get_raw_hsi_cube
from src.cnn.cnn_preprocessing import standardise_data


def predict(run_path, path, x, y):
    """Classifies every pixel of a map using the model in a trained run.

    Args:
        run_path (str | Path): A run directory written by training. Its manifest says
            how long a spectrum to build and over what wavenumber range, so those
            settings come from the model rather than from this file.
        path (str): Directory holding the ``FullGrid[n](...)`` spectra.
        x (int): Number of rows in the grid.
        y (int): Number of columns in the grid.

    Returns:
        tuple[np.ndarray, np.ndarray]: The most likely mineral per pixel, and the five
            most likely with their probabilities.

    Raises:
        FileNotFoundError: ``run_path`` is not a run directory.
        ValueError: The run's model and manifest disagree, or the spectra do not cover
            the range the model was trained on.
    """
    run = runs.load_run(run_path)
    manifest = run.manifest
    model = run.model
    le = run.encoder

    hsi_cube = get_raw_hsi_cube(path, x, y)
    runs.check_axis_covers_training_range(manifest, hsi_cube.spectral_axis)

    length, _ = hsi_cube.shape

    predicted_labels_map = []
    predicted_top5_map = []

    for l in range(length):
        # standardise each row of pixels for prediction
        spectra_list = list(hsi_cube[l])
        x_new = standardise_data(
            spectra_list,
            target_length=manifest.input_size,
            x_min=manifest.x_min,
            x_max=manifest.x_max,
        )
        predictions = model.predict(x_new, verbose=0)  # (width, num_classes)

        # top prediction per pixel
        predicted_indices = np.argmax(predictions, axis=1)
        row_labels = le.inverse_transform(predicted_indices)
        predicted_labels_map.append(row_labels)

        # top 5 predictions per pixel
        row_top5 = []
        for prob_vector in predictions:
            top5_indices = np.argsort(prob_vector)[::-1][:5]
            top5_labels = le.inverse_transform(top5_indices)
            top5_probs = prob_vector[top5_indices]
            row_top5.append(list(zip(top5_labels, top5_probs)))
        predicted_top5_map.append(row_top5)

    predicted_labels_map = np.array(predicted_labels_map)
    predicted_top5_map = np.array(predicted_top5_map, dtype=object)

    return predicted_labels_map, predicted_top5_map
