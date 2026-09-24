
from keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
from keras.utils import plot_model
from keras.callbacks import EarlyStopping
import pickle

from src.data.loader import *
from src.cnn.cnn_preprocessing import *
from src.cnn.model import *
from src.cnn.evaluate import *
from collections import Counter
from src import paths

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

with open("artifacts/encoders/label_encoder_final_poor2.pkl", "wb") as f:
    pickle.dump(le, f)

spectra_train = [spectra_list[i] for i in train_index]
spectra_test = [spectra_list[i] for i in test_index]

# standardise the training and testing data
x_train = standardise_data(spectra_train, target_length=input_size, x_min=x_min, x_max=x_max)
x_test = standardise_data(spectra_test, target_length=input_size, x_min=x_min, x_max=x_max)

x_all, y_all_labels = build_augmented_dataset(x_train, y_train_labels, 1, 1)

"""
# augment by summing possible linear combinations
x_combos, y_combos_labels = build_augmented_dataset(x_train, y_train_labels, n_combination=1)

# update training datasset to include augmented data
x_train = np.concatenate([x_train, x_combos], axis=0)
y_train_labels = np.concatenate([y_train_labels, y_combos_labels], axis=0)
"""

model = CNN_Model(num_classes, input_size)
model.summary()
plot_model(model, to_file=paths.output_path('model_plot_final_poor2.png'), show_shapes=True)

# create a class array of ints, since classes are labelled as string 
classes_array = np.arange(num_classes)
y_all_integers = le.transform(y_all_labels)
class_weights = compute_class_weight(class_weight='balanced', classes=classes_array, y=y_all_integers)
class_weight_dict = dict(enumerate(class_weights))

"""
train_gen = DataGenerator(
    x_all, y_all_integers,
    num_classes,
    input_size,
    batch_size=64,
    shuffle=True,
    augment=True
    )

val_gen = DataGenerator(
    x_test, le.transform(y_test_labels),
    num_classes,
    input_size,
    batch_size=64,
    shuffle=False,
    augment=False
    )
"""

y_all = to_categorical(y_all_integers, num_classes)
early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

history = model.fit(
    x_all, y_all,
    epochs=500,
    callbacks=[early_stop],
    validation_data=(x_test, y_test),
    class_weight=class_weight_dict,
    )

# save full model and wavenumber range
model.save("artifacts/models/raman_cnn_model_final_poor2.keras")
model.save_weights("artifacts/weights/test_final_poor2.weights.h5")
np.save("artifacts/metadata/wavenumber_range_final_poor2.npy", np.array([x_min, x_max]))

show_results(history, model, x_test, y_test, paths.output_path("training_results_final_poor2.png"))