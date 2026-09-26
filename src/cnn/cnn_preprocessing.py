import numpy as np
from collections import defaultdict
from scipy.interpolate import interp1d

def standardise_data(spectra_list, target_length, x_min=0, x_max=1400):
    standardised_data = []
    # new wavenumber axis the intensity will be interpolated to
    new_x = np.linspace(x_min, x_max, target_length)
    # get all standardised data
    for spectrum in spectra_list:
        x = spectrum.spectral_axis
        y = spectrum.spectral_data
        # interpolator
        f = interp1d(x, y, kind='linear', bounds_error=False, fill_value=0)
        # new interpolated intensities
        new_y = f(new_x)
        # clip the new_y to avoid intensity being below 0
        new_y = np.clip(new_y, 0, None)
        # normalise
        new_y /= (new_y.max() + 1e-8) # plus small value to avoid div by 0 error
        # append new y as a column as required by CNN model
        standardised_data.append(new_y.reshape(-1, 1))
    # returns shape (target_length, 1)
    return np.array(standardised_data)

def augment_shift(spectrum, rng):
    """A copy of the spectrum moved along the axis by a random number of points.

    Args:
        spectrum: One standardised spectrum, shaped (points, 1).
        rng (np.random.Generator): The run's generator. Passed in rather than drawn from
            ``np.random`` so that a run's augmentations depend on its seed and on nothing
            else that happens to be drawing from the global generator.
    """
    intensity = spectrum[:, 0].copy() # again, the shape is (target_length, 1)

    # generate some random shift integer val between -5 and 5
    shift = rng.integers(-5, 6)
    # shifts = 0 do nothing
    while shift == 0:
        shift = rng.integers(-5, 6)

    # use numpy roll to shift for its quickness
    intensity = np.roll(intensity, shift)
    # np roll wraps, replace values to 0 that were wrapped around
    if shift > 0:
        intensity[:shift] = 0
    else:
        intensity[shift:] = 0

    # reshape to fit CNN model's req
    spectrum = intensity.reshape(-1, 1)
    return spectrum

def augment_noise(spectrum, rng):
    """A copy of the spectrum with noise added in proportion to its own intensity."""
    intensity = spectrum[:, 0].copy()
    noise = rng.normal(0, 0.1 * np.abs(intensity))

    # clip to stay bounded between 0 and inf
    intensity = np.clip(intensity + noise, 0, None)
    # normalise
    intensity /= (intensity.max() + 1e-8)
    # reshape to fit CNN model's req
    spectrum = intensity.reshape(-1, 1)
    return spectrum

def augment_linear_combinations(spectra_for_class, rng):
    """Every pair of spectra in a class mixed by a random coefficient."""
    augmented = []
    n = len(spectra_for_class)
    if n < 2:
        return augmented

    # all unique pairs
    for i in range(n):
        for j in range(i + 1, n):
            coeff = rng.uniform(0, 1)
            combined_intensity = (
                coeff * spectra_for_class[i][:, 0] +
                (1 - coeff) * spectra_for_class[j][:, 0]
            )
            combined_intensity /= (combined_intensity.max() + 1e-8)
            augmented.append(combined_intensity.reshape(-1, 1))
    return augmented

def build_augmented_dataset(x_train, y_train_labels, n_shift, n_noise, rng):
    """The originals plus their augmented copies, and their labels.

    Augmentation is applied to the originals only, never to an augmented spectrum, so a
    run's augmented set is ``n_shift + n_noise`` copies plus one combination per pair of
    spectra within a class.
    """

    original_x  = list(x_train)
    original_y  = list(y_train_labels)
    augmented_x = []
    augmented_y = []

    # collect the spectras grouped by classes
    # running hash map appraoch
    class_spectra = {}
    for spectrum, label in zip(x_train, y_train_labels):
        if label not in class_spectra:
            class_spectra[label] = []
        class_spectra[label].append(spectrum)

    for label, spectra_list in class_spectra.items():

        for spectrum in spectra_list:

            # shift augmentation on originals only
            for _ in range(n_shift):
                augmented_x.append(augment_shift(spectrum, rng))
                augmented_y.append(label)

            # noise augmentation on originals only
            for _ in range(n_noise):
                augmented_x.append(augment_noise(spectrum, rng))
                augmented_y.append(label)

        # augment by linear combinations
        combos = augment_linear_combinations(spectra_list, rng)
        for combo in combos:
            augmented_x.append(combo)
            augmented_y.append(label)


    # originals + augmented, never augmenting augmented data
    x_all = np.array(original_x + augmented_x)
    y_all = np.array(original_y + augmented_y)

    print(f"Original samples:            {len(original_x)}")
    print(f"Shift augmented:             {sum(1 for _ in range(n_shift)) * len(x_train)}")
    print(f"Noise augmented:             {sum(1 for _ in range(n_noise)) * len(x_train)}")
    print(f"Linear combinations:         {len(augmented_x) - n_shift * len(x_train) - n_noise * len(x_train)}")
    print(f"Total augmented samples:     {len(augmented_x)}")
    print(f"Total samples:               {len(x_all)}")
    print(f"Augmentation multiplier:     {len(x_all) / len(original_x):.1f}x")

    return x_all, y_all

def leave_one_out_split(y_data, rng):
    """The indices to train on and the indices held out, one spectrum per class.

    A class with a single spectrum stays in the training set, since holding it out would
    leave nothing to train that class on. Which spectrum is held out for the others is
    drawn from ``rng``, so the split is a property of the run's seed: it decides which
    spectra the reported accuracy was measured on.
    """
    # since dictionary can not hold duplicate keys, use that to hold index
    class_indices = defaultdict(list)
    for i, label in enumerate(y_data):
        class_indices[label].append(i)

    train_index = []
    test_index = []

    for label, indices in class_indices.items():
        if len(indices) == 1:
            train_index.extend(indices)
        else:
            test_pick = rng.choice(np.asarray(indices))
            test_index.append(test_pick)
            train_index.extend([i for i in indices if i != test_pick])

    return np.array(train_index), np.array(test_index)
