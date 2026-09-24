"""Settings that used to be literals inside the module that happened to need them.

Keeping a number next to the code that reads it is fine until a second caller wants
a different one, or until you need to know what a run actually used. Each config
here is a frozen dataclass with defaults that reproduce the behaviour that was
inlined, so constructing the default object changes nothing.
"""

from dataclasses import dataclass, fields


@dataclass(frozen=True)
class NmfConfig:
    """The settings for an NMF decomposition of a preprocessed cube.

    The component count is an argument to :meth:`as_kwargs` rather than a field,
    because it is decided per call: it comes from the caller, or from the phase
    estimator when the caller does not know it.

    ``random_state`` is ``None`` here, which is what the decomposition used when the
    settings were inline. It is not the source of run-to-run variation in practice,
    because the ``nndsvda`` initialisation is deterministic; the seed that matters is
    the one the split and the augmentation draw from.
    """

    init: str = "nndsvda"
    solver: str = "cd"
    beta_loss: str = "frobenius"
    tol: float = 1e-4
    max_iter: int = 10000
    random_state: int | None = None
    alpha_W: float = 0.0
    alpha_H: float | str = "same"
    l1_ratio: float = 0.0
    verbose: int = 0
    shuffle: bool = False

    def as_kwargs(self, n_components: int) -> dict:
        """The keyword arguments for ``ramanspy``'s ``NMF``, component count included."""
        settings = {field.name: getattr(self, field.name) for field in fields(self)}
        return {"n_components": n_components, **settings}


@dataclass(frozen=True)
class EstimatorConfig:
    """The bounds that decide whether a phase-count estimate is usable.

    These were literals inside the combination rule, which meant the rule could not
    be read in one place and a caller could not loosen or tighten it. Each bound is
    exclusive at its own limit, so the value named here is still kept.

    Attributes:
        reject_above: An estimate larger than this is rejected. On a real cube the
            PCA methods can return a count in the tens, which no sample of this kind
            has, so anything past this is treated as a failure of the method rather
            than as an answer.
        min_votes: How many estimates have to survive before a count is named at all.
            Below this the result is undetermined, because the largest of one or two
            numbers is not something to decompose a cube on.
        hfc_tolerance: How close the NW-HFC estimate has to be to the largest PCA
            estimate to count as agreeing with it.
    """

    reject_above: int = 10
    min_votes: int = 2
    hfc_tolerance: int = 2


@dataclass(frozen=True)
class CnnConfig:
    """The shape and the optimiser of the convolutional model.

    The defaults are the architecture that was hardcoded in ``CNN_Model``: three
    convolutional blocks with filters of 16, 32 and 64 and kernels of 21, 11 and 5,
    then a 2048-unit dense layer. Changing them here changes the model, which is the
    point of the config existing, but any change to the defaults changes the
    parameter count and so the model that a training run produces.

    ``initializer_variance`` is a variance, and the initialiser is built with its
    square root as the standard deviation.
    """

    conv_filters: tuple[int, ...] = (16, 32, 64)
    conv_kernel_sizes: tuple[int, ...] = (21, 11, 5)
    pool_size: int = 2
    dense_units: int = 2048
    dropout: float = 0.5
    initializer_variance: float = 0.05
    learning_rate: float = 0.001
    beta_1: float = 0.9
    beta_2: float = 0.999
    epsilon: float = 1e-08


@dataclass(frozen=True)
class TrainConfig:
    """What one training run needs, and nothing about where the process was started.

    The dataset and the wavenumber range used to be literals in the middle of the
    training script, so a second dataset meant editing the script. They are fields
    now, and the command line reads its defaults from them, so the two cannot drift
    apart: ``--input-size`` with no argument offers ``input_size``.
    """

    dataset: str = "~/Code/Data_SH/poor_unoriented"
    input_size: int = 2000
    x_min: float = 313.3
    x_max: float = 1028.4
    epochs: int = 500
    patience: int = 20
    n_shift: int = 1
    n_noise: int = 1
    name: str | None = None

