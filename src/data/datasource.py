from dataclasses import dataclass, field


@dataclass
class DataSource:
    start: str | None = None
    end: str | None = None
    csv: str | None = None
    netcdf_1d: str | None = None
    netcdf_1d_vars: list = field(default_factory=list)
    csv_index_col: int = 0
    binary_cols: list = field(default_factory=list)  # columns to skip normalization for; empty = auto-detect
    nodata_values: list = field(default_factory=list)  # sentinel values (e.g. -999) treated as missing, across all columns

    def __post_init__(self):
        if not (self.csv or self.netcdf_1d):
            raise ValueError("At least one of csv or netcdf_1d must be set")
        if self.netcdf_1d and not self.netcdf_1d_vars:
            raise ValueError("netcdf_1d_vars is required when netcdf_1d is set")
