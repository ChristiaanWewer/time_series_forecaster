import numpy as np
import xarray as xr


class ZarrStoreManager:
    _open_stores: dict[str, xr.Dataset] = {}

    @classmethod
    def open(cls, path: str) -> xr.Dataset:
        if path not in cls._open_stores:
            cls._open_stores[path] = xr.open_zarr(path)
        return cls._open_stores[path]

    @staticmethod
    def coord_to_index(
        ds: xr.Dataset,
        location: tuple,
        coord_dims: tuple,
        location_is_index: bool,
    ) -> tuple[int, int]:
        if location_is_index:
            return int(location[0]), int(location[1])
        cx, cy = location
        dim_x, dim_y = coord_dims
        ix = int(np.abs(ds[dim_x].values - cx).argmin())
        iy = int(np.abs(ds[dim_y].values - cy).argmin())
        return ix, iy

    @staticmethod
    def read_patch(
        ds: xr.Dataset,
        vars: list[str],
        time_indices: list[int],
        ix: int,
        iy: int,
        patch_size: int,
        coord_dims: tuple,
    ) -> dict[str, np.ndarray]:
        half = patch_size // 2
        dim_x, dim_y = coord_dims
        spatial_sel = {
            dim_x: slice(ix - half, ix + half + 1),
            dim_y: slice(iy - half, iy + half + 1),
        }
        result = {}
        for v in vars:
            result[v] = ds[v].isel(time=time_indices, **spatial_sel).values
        return result

    @staticmethod
    def read_static_patch(
        ds: xr.Dataset,
        vars: list[str],
        ix: int,
        iy: int,
        patch_size: int,
        coord_dims: tuple,
    ) -> dict[str, np.ndarray]:
        half = patch_size // 2
        dim_x, dim_y = coord_dims
        spatial_sel = {
            dim_x: slice(ix - half, ix + half + 1),
            dim_y: slice(iy - half, iy + half + 1),
        }
        result = {}
        for v in vars:
            result[v] = ds[v].isel(**spatial_sel).values
        return result
