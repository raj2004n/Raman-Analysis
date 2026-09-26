import os
import numpy as np
import pandas as pd
import ramanspy as rp
from dataclasses import dataclass
from .grid import RamanGrid
from scipy.integrate import cumulative_trapezoid


@dataclass(frozen=True)
class Cube:
    """A cube of values per pixel together with the arrays that describe it.

    ``get_cumulative_hsi_cube`` used to return these four as a tuple, which callers
    unpacked by position. A tuple says nothing about the order, so a change to the order
    was invisible at every call site, and a viewer that needed only the axis still had to
    name the other three.

    Attributes:
        cube: The values, shaped (rows, cols, bands).
        spectra: One row per pixel holding the preprocessed spectrum the cube was built
            from, shaped (rows * cols + 1, bands). Row 0 is unused, because pixel
            numbering starts at 1.
        axis: The wavenumber axis the bands sit on, after any cropping, shaped (bands,).
        pixel_map: The pixel number at each position, shaped (rows, cols), which is what
            turns a click on an image back into a spectrum.
    """

    cube: np.ndarray
    spectra: np.ndarray
    axis: np.ndarray
    pixel_map: np.ndarray


def load_CNN_data(path="~/Code/Data_SH/poor_unoriented"):
    """Loads Raman Spectra dataset from RRUFF for training the CNN model.

    The spectra come back in the order the files are read, which is the order
    ``glob`` returns them in rather than a sorted one. That order reaches the
    training split, since the split is over positions rather than over names, so a
    replay has to reproduce it. The run record keeps a digest of it for that reason,
    and ``runs.dataset_order_digest`` computes the same list the same way.

    Args:
        path (str, optional): File path for the dataset.
            Defaults to "~/Code/Data_SH/poor_unoriented".

    Returns:
        SpectralContainer, List[dict]: RamanSPy spectral container containing spectra and metadata
    """
    path = os.path.expanduser(path)
    spectra, metadata = rp.datasets.rruff(path, download=False)
    return spectra, metadata

def read_spectrum(file):
    """Reads the spectrum from the passed in file.

    Args:
        file : _description_

    Returns:
        _type_: _description_
    """
    df = pd.read_csv(file, sep='\t', names=['raman_shift', 'intensity'], header=None, usecols=[0, 1])    
    return df['raman_shift'].tolist(), df['intensity'].values

def build_pipeline(pipeline_id, normalisation_pixelwise=True, fingerprint=False):
    if pipeline_id == 0:
        return None
    elif pipeline_id == 1:
        return rp.preprocessing.protocols.georgiev2023_P1(normalisation_pixelwise, fingerprint)
    elif pipeline_id == 2:
        return rp.preprocessing.protocols.georgiev2023_P2(normalisation_pixelwise, fingerprint)
    elif pipeline_id == 3:
        return rp.preprocessing.protocols.georgiev2023_P3(normalisation_pixelwise, fingerprint)
    else:
        raise ValueError(f"Unknown pipeline id: {pipeline_id}")
    
def crop_axis(spectral_axis, start=None, end=None):
    spectral_axis = np.array(spectral_axis)
    start_idx = np.searchsorted(spectral_axis, start) if start is not None else 0
    end_idx = np.searchsorted(spectral_axis, end) if end is not None else len(spectral_axis)
    return spectral_axis[start_idx:end_idx], start_idx, end_idx

def get_raw_hsi_cube(path, x, y, start=None, end=None):
    grid = RamanGrid(path, x, y)

    # read spectral axis from first file only, same across all so its fine
    files = grid.get_sorted_files()
    spectral_axis, _ = read_spectrum(files[0])

    # crop spectral axis and get start and end indexes of cropping region
    spectral_axis, start_idx, end_idx = crop_axis(spectral_axis, start, end)

    spectral_data = np.zeros((x, y, len(spectral_axis)))

    for file, row, col in grid.traverse():
        _, intensity = read_spectrum(file)
        # crop spectral_data to match spectral_axis, assign to spectral_data
        spectral_data[row, col] = intensity[start_idx:end_idx]
    
    return rp.SpectralImage(spectral_data, spectral_axis)

def get_cumulative_hsi_cube(path, x, y, pipeline_id, start=None, end=None):
    """Loads the grid, preprocesses each pixel, and integrates each spectrum.

    The cube holds the cumulative trapezoidal integral of every preprocessed spectrum,
    so a band of it is the area under the spectrum up to that wavenumber, and the
    difference between two bands is the area between them.

    Args:
        path: Directory holding the .txt spectra.
        x, y: Rows and columns of the grid.
        pipeline_id: Which preprocessing protocol to apply, 0 for none.
        start, end: Wavenumber bounds to crop to. Either may be None.

    Returns:
        Cube: the cumulative cube, the spectra behind it, the cropped axis, and the
        pixel number at each position.
    """
    # grid object to traverse through the grid
    grid = RamanGrid(path, x, y)

    # build pipeline and cropper
    pipeline = build_pipeline(pipeline_id)
    cropper = rp.preprocessing.misc.Cropper(region=(start, end))

    # get lenght of cropped spectral axis
    files = grid.get_sorted_files()
    spectral_axis, _ = read_spectrum(files[0])
    ref_spectrum = rp.Spectrum(np.zeros(len(spectral_axis)), spectral_axis) # reference spectrum
    ref_spectrum = cropper.apply(ref_spectrum) # crop reference spectrum
    cropped_axis = ref_spectrum.spectral_axis
    n = len(cropped_axis) # length of cropped spectral axis

    # initialise variables
    cumulative_cube = np.zeros((x, y, n)) # will hold the cumulative area
    spectra_list = np.zeros((x * y + 1, n)) # will hold the spectrum for each pixel
    pixel_map = np.zeros((x, y), dtype=int) # need this later for plotting

    # initial start at pixel 1
    pixel = 1 
    # traverse through full grid of data
    for file, row, col in grid.traverse():
        # get and prepare spectrum
        spectral_axis, spectral_data = read_spectrum(file)
        spectrum = rp.Spectrum(spectral_data, spectral_axis)
        spectrum = cropper.apply(spectrum)
        if pipeline is not None:
            spectrum = pipeline.apply(spectrum)

        # store spectral data
        spectral_data = spectrum.spectral_data # intensities
        spectra_list[pixel] = spectral_data

        # store the cumulative sum using trapezoidal integration
        cum = cumulative_trapezoid(spectral_data, cropped_axis, initial=0) # returns n-1 vals so prepend 0 to get n vals
        cumulative_cube[row, col] = cum

        # store pixel and update
        pixel_map[row, col] = pixel
        pixel += 1

    return Cube(cumulative_cube, spectra_list, cropped_axis, pixel_map)
