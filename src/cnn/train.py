"""Trains the Raman CNN and writes one run directory for prediction to load.

The training flow is a function of a ``TrainConfig`` rather than module level code, so
importing this module does not train anything and a second dataset is a different
config rather than an edit. The command line entry point is ``main``, and its defaults
come from the same config, so the documented defaults and the code cannot disagree.
"""

import argparse

import numpy as np
from keras.callbacks import EarlyStopping
from keras.utils import plot_model, set_random_seed, to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

from src.config import TrainConfig
from src.cnn import runs
from src.cnn.cnn_preprocessing import build_augmented_dataset, leave_one_out_split, standardise_data
from src.cnn.evaluate import show_results
from src.cnn.model import CNN_Model
from src.data.loader import load_CNN_data


def train(cfg: TrainConfig) -> runs.RunManifest:
    """Runs one training job and returns the manifest written for it.

    One class is held out per class by ``leave_one_out_split``, the rest are standardised
    onto ``cfg.input_size`` points between ``cfg.x_min`` and ``cfg.x_max``, and the
    training half is augmented before fitting. The run directory records the input size
    and the wavenumber range, because a prediction has to reproduce both, and it records
    the seed, the dataset digests and the versions the run used, so the accuracy it
    reports can be traced back to a run that can be repeated.
    """
    run_name = cfg.name or runs.new_run_name()
    run_dir = runs.run_directory(run_name)

    # The model plot is written into the run directory before the manifest is, so the
    # directory has to exist from the start. A run under a name that has not been used
    # before is the ordinary case, and creating the directory only when the manifest is
    # saved left that case failing on the plot with no such file or directory.
    run_dir.mkdir(parents=True, exist_ok=True)

    # The one source of randomness for the data side: this generator decides which
    # spectrum each class holds out and how the rest are augmented. It is a local
    # generator rather than a seed set on the global one, so the run is repeatable from
    # its config no matter what else in the process draws random numbers.
    rng = np.random.default_rng(cfg.seed)

    spectra_list, metadata = load_CNN_data(cfg.dataset)

    y_data = [m['##NAMES'].split(',')[0].strip() for m in metadata]

    train_index, test_index = leave_one_out_split(y_data, rng)

    # one hot encoding
    le = LabelEncoder()
    le.fit(y_data)

    classes = le.classes_
    num_classes = len(classes)

    y_data = np.array(y_data)

    y_train_labels = y_data[train_index]
    y_test_labels = y_data[test_index]

    y_test = to_categorical(le.transform(y_test_labels), num_classes)

    spectra_train = [spectra_list[i] for i in train_index]
    spectra_test = [spectra_list[i] for i in test_index]

    # standardise the training and testing data
    x_train = standardise_data(spectra_train, target_length=cfg.input_size, x_min=cfg.x_min, x_max=cfg.x_max)
    x_test = standardise_data(spectra_test, target_length=cfg.input_size, x_min=cfg.x_min, x_max=cfg.x_max)

    x_all, y_all_labels = build_augmented_dataset(x_train, y_train_labels, cfg.n_shift, cfg.n_noise, rng)

    # The network's initial weights are drawn when the layers are built, and Keras takes
    # those draws from its own global state, so this is the one setting a run cannot pass
    # down as an argument. Seeding it here, immediately before the model is built, is what
    # makes two runs of the same config produce the same weights.
    set_random_seed(cfg.seed)
    model = CNN_Model(num_classes, cfg.input_size)
    model.summary()
    plot_model(model, to_file=run_dir / runs.MODEL_PLOT_NAME, show_shapes=True)

    # create a class array of ints, since classes are labelled as string
    classes_array = np.arange(num_classes)
    y_all_integers = le.transform(y_all_labels)
    class_weights = compute_class_weight(class_weight='balanced', classes=classes_array, y=y_all_integers)
    class_weight_dict = dict(enumerate(class_weights))

    y_all = to_categorical(y_all_integers, num_classes)
    early_stop = EarlyStopping(monitor='val_loss', patience=cfg.patience, restore_best_weights=True)

    history = model.fit(
        x_all, y_all,
        epochs=cfg.epochs,
        callbacks=[early_stop],
        validation_data=(x_test, y_test),
        class_weight=class_weight_dict,
        )

    # What the fitted model scores on the spectra that were held out. This is the
    # number a report quotes, so it is measured here and written into the run's own
    # record, where it stays attached to the seed and the data that produced it.
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Held-out accuracy: {test_accuracy:.4f} over {len(y_test)} spectra")

    # one directory per run, so the settings, the seed and the dataset travel with the
    # model instead of having to be remembered when predicting or when defending a number
    manifest = runs.save_run(
        run_dir,
        config=cfg,
        model=model,
        encoder=le,
        metrics={"test_loss": float(test_loss), "test_accuracy": float(test_accuracy)},
    )
    print(f"Wrote run '{manifest.name}' to {run_dir}")

    show_results(history, model, x_test, y_test, run_dir / runs.TRAINING_CURVES_NAME)

    return manifest


def parse_args(argv=None):
    """Reads the command line, taking every default from ``TrainConfig``."""
    parser = argparse.ArgumentParser(
        description="Train the Raman CNN and write one run directory for prediction to load."
    )
    parser.add_argument(
        "--name",
        default=TrainConfig.name,
        help="name for the run directory; used when predicting. Defaults to a timestamp.",
    )
    parser.add_argument(
        "--dataset",
        default=TrainConfig.dataset,
        help="directory of .txt spectra to train on.",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=TrainConfig.input_size,
        help="number of points on the standardised wavenumber axis.",
    )
    parser.add_argument("--x-min", type=float, default=TrainConfig.x_min, help="low end of the axis, in cm-1.")
    parser.add_argument("--x-max", type=float, default=TrainConfig.x_max, help="high end of the axis, in cm-1.")
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs, help="epoch cap for the fit.")
    parser.add_argument(
        "--patience",
        type=int,
        default=TrainConfig.patience,
        help="epochs without an improvement in validation loss before stopping.",
    )
    parser.add_argument("--n-shift", type=int, default=TrainConfig.n_shift, help="shift augmentations per spectrum.")
    parser.add_argument("--n-noise", type=int, default=TrainConfig.n_noise, help="noise augmentations per spectrum.")
    parser.add_argument(
        "--seed",
        type=int,
        default=TrainConfig.seed,
        help="seed for the held-out split, the augmentations and the initial weights.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    """Trains one run from the command line arguments."""
    args = parse_args(argv)
    return train(TrainConfig(
        dataset=args.dataset,
        input_size=args.input_size,
        x_min=args.x_min,
        x_max=args.x_max,
        epochs=args.epochs,
        patience=args.patience,
        n_shift=args.n_shift,
        n_noise=args.n_noise,
        seed=args.seed,
        name=args.name,
    ))


if __name__ == "__main__":
    main()
