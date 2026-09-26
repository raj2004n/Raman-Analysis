"""The training curves, and the score the run was measured at.

``show_results`` plots the history a fit returned and logs the two numbers beside it. The
numbers are handed in rather than measured here: the training flow already evaluates the
model on the held-out spectra to write the score into the run's record, and measuring it a
second time here would mean the figure and the record could disagree about the same run.
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def show_results(history, metrics, save_path):
    """Writes the accuracy and loss curves to ``save_path`` and shows them.

    Args:
        history: What ``model.fit`` returned, whose ``history`` mapping holds one entry
            per epoch for 'accuracy', 'val_accuracy', 'loss' and 'val_loss'.
        metrics (dict): The scores the run was measured at, keyed 'test_loss' and
            'test_accuracy'. These are the numbers the run's record carries.
        save_path: Where to write the figure.
    """
    logger.info("Test loss: %s", metrics["test_loss"])
    logger.info("Test accuracy: %s", metrics["test_accuracy"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy plot
    ax1.plot(history.history['accuracy'], label='training')
    ax1.plot(history.history['val_accuracy'], label='validation')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend(loc='best')

    # Loss plot
    ax2.plot(history.history['loss'], label='training')
    ax2.plot(history.history['val_loss'], label='validation')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend(loc='best')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
