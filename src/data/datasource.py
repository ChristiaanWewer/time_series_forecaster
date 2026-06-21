from dataclasses import dataclass, field


@dataclass
class DataSource:
    start: str
    end: str
    csv: str | None = None
    netcdf_1d: str | None = None
    netcdf_1d_vars: list = field(default_factory=list)
    zarr: str | None = None
    location: tuple | None = None
    location_is_index: bool = True
    coord_dims: tuple = ('x', 'y')
    target_col: str = 'target'
    static_zarr_vars: list = field(default_factory=list)
    csv_index_col: int = 0

    def __post_init__(self):
        if not (self.csv or self.netcdf_1d or self.zarr):
            raise ValueError("At least one of csv, netcdf_1d, or zarr must be set")
        if self.zarr and self.location is None:
            raise ValueError("location is required when zarr is set")
        if self.netcdf_1d and not self.netcdf_1d_vars:
            raise ValueError("netcdf_1d_vars is required when netcdf_1d is set")
