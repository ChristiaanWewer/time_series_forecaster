import torch
from torch.utils.data import DataLoader


class _CudaPrefetcher:
    """Wraps a DataLoader and copies batch n+1 to the GPU on a side CUDA stream while
    batch n is still computing, so the GPU never waits on data."""

    def __init__(self, loader: DataLoader, device: torch.device):
        """Require a pinned-memory loader (otherwise the copy silently becomes
        synchronous), create the side stream and start copying the first batch."""
        if not loader.pin_memory:
            raise ValueError(
                '_CudaPrefetcher needs a DataLoader with pin_memory=True; without pinned '
                'memory non_blocking copies silently become synchronous.'
            )
        self._iterator = iter(loader)
        self._device = device
        self._stream = torch.cuda.Stream(device=device)
        self._next = None
        self._preload()

    def __iter__(self) -> '_CudaPrefetcher':
        """Return itself; a fresh prefetcher is made for each epoch."""
        return self

    def __next__(self) -> dict:
        """Wait for the prefetched batch's copy, hand it over, mark its tensors as used on
        the compute stream (so their memory is not reused too early) and start copying
        the next one."""
        if self._next is None:
            raise StopIteration
        compute_stream = torch.cuda.current_stream(self._device)
        compute_stream.wait_stream(self._stream)
        batch = self._next
        for tensor in batch.values():
            tensor.record_stream(compute_stream)
        self._preload()
        return batch

    def _preload(self) -> None:
        """Take the next batch from the loader and start its non-blocking copy on the side
        stream, or remember that the loader is exhausted."""
        try:
            batch = next(self._iterator)
        except StopIteration:
            self._next = None
            return
        with torch.cuda.stream(self._stream):
            self._next = {k: v.to(self._device, non_blocking=True) for k, v in batch.items()}


def _make_prefetcher(loader: DataLoader, device: torch.device):
    """The only entry point the Forecaster uses. Returns a _CudaPrefetcher on CUDA with a
    pinned loader, and otherwise a plain generator that moves each batch to the device.
    Both yield the same dicts of on-device tensors."""
    device = torch.device(device)
    if device.type == 'cuda' and loader.pin_memory:
        return _CudaPrefetcher(loader, device)
    return ({k: v.to(device) for k, v in batch.items()} for batch in loader)
