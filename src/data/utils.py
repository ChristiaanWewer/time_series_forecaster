import xarray as xr


def netcdf_to_zarr(input_path: str, output_path: str, chunk_sizes: dict):
    ds = xr.open_dataset(input_path)
    ds.chunk(chunk_sizes).to_zarr(output_path, mode='w')
