import numpy as np
import jax.numpy as jnp

class DataLoader:
    """
    A simple DataLoader for JAX that mimics torch.utils.data.DataLoader,
    padding the dataset once at initialization so every batch is full.
    Expects `dataset` to be a tuple of JAX arrays (all with same first dim).
    """
    def __init__(
        self,
        dataset,
        batch_size: int = 1,
        shuffle: bool = False,
        pad_value: float = 0.0
    ):
        self.batch_size = batch_size
        self.shuffle = shuffle

        # original dataset arrays
        arrays = list(dataset)
        orig_size = arrays[0].shape[0]
        # compute total size after padding
        pad_size = (-orig_size) % batch_size  # 0 if already divisible
        new_size = orig_size + pad_size

        # pad each array along axis=0 to new_size
        if pad_size > 0:
            padded = []
            for arr in arrays:
                pad_cfg = [(0, pad_size)] + [(0, 0)] * (arr.ndim - 1)
                padded_arr = jnp.pad(arr, pad_cfg, constant_values=pad_value)
                padded.append(padded_arr)
            self.dataset = tuple(padded)
        else:
            self.dataset = tuple(arrays)

        self.dataset_size = new_size
        self.num_batches = new_size // batch_size

    def __iter__(self):
        # generate indices for the (padded) dataset
        idxs = np.arange(self.dataset_size)
        if self.shuffle:
            np.random.shuffle(idxs)

        # yield consecutive batches
        for i in range(self.num_batches):
            batch_idxs = idxs[i * self.batch_size : (i + 1) * self.batch_size]
            yield tuple(jnp.take(arr, batch_idxs, axis=0) for arr in self.dataset)

    def __len__(self):
        return self.num_batches
