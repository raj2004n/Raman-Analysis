
import argparse

from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from keras.utils import plot_model
from keras.callbacks import EarlyStopping

from src.data.loader import *
from src.cnn.cnn_preprocessing import *
from src.cnn.model import *
from src.cnn.evaluate import *
from src.cnn import runs

parser = argparse.ArgumentParser(
    description="Train the Raman CNN and write one run directory for prediction to load."
)
parser.add_argument(
    "--name",
    default=None,
    help="name for the run directory; used when predicting. Defaults to a timestamp.",
)
args = parser.parse_args()

run_name = args.name or runs.new_run_name()
run_dir = runs.run_directory(run_name)

path="~/Code/Data_SH/poor_unoriented"
input_size = 2000
# hard coded for now
x_min, x_max = 313.3, 1028.4

spectra_list, metadata = load_CNN_data(path)

y_data = [m['##NAMES'].split(',')[0].strip() for m in metadata]

train_index, test_index = leave_one_out_split(y_data)

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
x_train = standardise_data(spectra_train, target_length=input_size, x_min=x_min, x_max=x_max)
x_test = standardise_data(spectra_test, target_length=input_size, x_min=x_min, x_max=x_max)

x_all, y_all_labels = build_augmented_dataset(x_train, y_train_labels, 1, 1)

model = CNN_Model(num_classes, input_size)
model.summary()
plot_model(model, to_file=run_dir / runs.MODEL_PLOT_NAME, show_shapes=True)

# create a class array of ints, since classes are labelled as string 
classes_array = np.arange(num_classes)
y_all_integers = le.transform(y_all_labels)
class_weights = compute_class_weight(class_weight='balanced', classes=classes_array, y=y_all_integers)
class_weight_dict = dict(enumerate(class_weights))

y_all = to_categorical(y_all_integers, num_classes)
early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

history = model.fit(
    x_all, y_all,
    epochs=500,
    callbacks=[early_stop],
    validation_data=(x_test, y_test),
    class_weight=class_weight_dict,
    )

# one directory per run, so the input length and the wavenumber range travel with
# the model instead of having to be remembered when predicting
manifest = runs.save_run(
    run_dir,
    name=run_name,
    model=model,
    encoder=le,
    input_size=input_size,
    num_classes=num_classes,
    x_min=x_min,
    x_max=x_max,
)
print(f"Wrote run '{manifest.name}' to {run_dir}")

show_results(history, model, x_test, y_test, run_dir / runs.TRAINING_CURVES_NAME)