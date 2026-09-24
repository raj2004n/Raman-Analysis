import matplotlib.pyplot as plt

def show_results(history, model, x_test, y_test, save_path):
    loss, accuracy = model.evaluate(x_test, y_test, verbose=2)
    print("Test loss:", loss)
    print("Test accuracy:", accuracy)

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