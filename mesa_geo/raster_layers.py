"""
Raster Layers
-------------
"""

from __future__ import annotations

import copy
import inspect
import itertools
import math
import warnings
from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import Any, cast, overload

import numpy as np
import rasterio as rio
from affine import Affine
from mesa import Model
from mesa.agent import Agent
from mesa.space import Coordinate, FloatCoordinate, accept_tuple_argument
from rasterio.warp import (
    Resampling,
    calculate_default_transform,
    reproject,
    transform_bounds,
)

from mesa_geo.geo_base import GeoBase


class RasterBase(GeoBase):
    """
    Base class for raster layers.
    """

    _width: int
    _height: int
    _transform: Affine
    _total_bounds: np.ndarray  # [min_x, min_y, max_x, max_y]

    def __init__(self, width, height, crs, total_bounds):
        """
        Initialize a raster base layer.

        :param width: Width of the raster base layer.
        :param height: Height of the raster base layer.
        :param crs: Coordinate reference system of the raster base layer.
        :param total_bounds: Bounds of the raster base layer in [min_x, min_y, max_x, max_y] format.
        """

        super().__init__(crs)
        self._width = width
        self._height = height
        self._total_bounds = total_bounds
        self._update_transform()

    @property
    def width(self) -> int:
        """
        Return the width of the raster base layer.

        :return: Width of the raster base layer.
        :rtype: int
        """

        return self._width

    @width.setter
    def width(self, width: int) -> None:
        """
        Set the width of the raster base layer.

        :param int width: Width of the raster base layer.
        """

        self._width = width
        self._update_transform()

    @property
    def height(self) -> int:
        """
        Return the height of the raster base layer.

        :return: Height of the raster base layer.
        :rtype: int
        """

        return self._height

    @height.setter
    def height(self, height: int) -> None:
        """
        Set the height of the raster base layer.

        :param int height: Height of the raster base layer.
        """

        self._height = height
        self._update_transform()

    @property
    def total_bounds(self) -> np.ndarray | None:
        """
        Return the bounds of the raster layer in [min_x, min_y, max_x, max_y] format.

        :return: Bounds of the raster layer in [min_x, min_y, max_x, max_y] format.
        :rtype: np.ndarray | None
        """

        return self._total_bounds

    @total_bounds.setter
    def total_bounds(self, total_bounds: np.ndarray) -> None:
        """
        Set the bounds of the raster base layer in [min_x, min_y, max_x, max_y] format.

        :param np.ndarray total_bounds: Bounds of the raster base layer in [min_x, min_y, max_x, max_y] format.
        """

        self._total_bounds = total_bounds
        self._update_transform()

    @property
    def transform(self) -> Affine:
        """
        Return the affine transformation of the raster base layer.

        :return: Affine transformation of the raster base layer.
        :rtype: Affine
        """

        return self._transform

    @property
    def resolution(self) -> tuple[float, float]:
        """
        Returns the (width, height) of a cell in the units of CRS.

        :return: Width and height of a cell in the units of CRS.
        :rtype: Tuple[float, float]
        """

        a, b, _, d, e, _, _, _, _ = self.transform
        return math.sqrt(a**2 + d**2), math.sqrt(b**2 + e**2)

    def _update_transform(self) -> None:
        self._transform = rio.transform.from_bounds(
            *self.total_bounds, width=self.width, height=self.height
        )

    def to_crs(self, crs, inplace=False) -> RasterBase | None:
        raise NotImplementedError

    def out_of_bounds(self, pos: Coordinate) -> bool:
        """
        Determines whether position is off the grid.

        :param Coordinate pos: Position to check.
        :return: True if position is off the grid, False otherwise.
        :rtype: bool
        """

        x, y = pos
        return x < 0 or x >= self.width or y < 0 or y >= self.height


class Cell(Agent):
    """
    Cells are containers of raster attributes, and are building blocks of `RasterLayer`.

    Deprecated:
        `Cell.indices` is deprecated. Use `Cell.rowcol` instead.
    """

    _pos: Coordinate | None
    _rowcol: Coordinate | None
    _xy: FloatCoordinate | None

    def __init__(
        self,
        model,
        pos=None,
        indices=None,
        *,
        rowcol=None,
        xy=None,
    ):
        """
        Initialize a cell.

        :param pos: Grid position of the cell in (grid_x, grid_y) format.
            Origin is at lower left corner of the grid
        :param indices: (Deprecated) Indices of the cell in (row, col) format.
            Origin is at upper left corner of the grid. Use rowcol instead.
        :param rowcol: Indices of the cell in (row, col) format.
            Origin is at upper left corner of the grid
        :param xy: Geographic/projected (x, y) coordinates of the cell center in the CRS.
        """

        super().__init__(model)
        self._pos = pos
        self._rowcol = indices if rowcol is None else rowcol
        self._xy = xy

    @property
    def pos(self) -> Coordinate | None:
        """
        Grid position in (grid_x, grid_y) format with origin at lower left of the grid.
        """
        return self._pos

    @pos.setter
    def pos(self, pos: Coordinate | None) -> None:
        """
        Deprecated setter for `pos`.
        """
        # mesa Agent set pos to None by default
        # avoid raising a warning when pos is set to None by the Agent constructor
        if pos is not None:
            warnings.warn(
                "Cell.pos setter is deprecated and will be read-only in a future release.",
                DeprecationWarning,
                stacklevel=2,
            )

        # set the pos for backward compatibility
        # in the future, this will be removed because pos is read-only
        self._pos = pos

    @property
    def indices(self) -> Coordinate | None:
        """
        Deprecated alias of `rowcol`.
        """
        warnings.warn(
            "Cell.indices is deprecated and will be removed in a future release. "
            "Use Cell.rowcol instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._rowcol

    @indices.setter
    def indices(self, indices: Coordinate | None) -> None:
        """
        Deprecated setter for `rowcol`.
        """
        warnings.warn(
            "Cell.indices is deprecated and will be removed in a future release. "
            "Use Cell.rowcol instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        # for backward compatibility, set the rowcol to the indices
        # in the future, this will be removed
        # and raise an AttributeError, because indices is read-only
        self._rowcol = indices

    @property
    def rowcol(self) -> Coordinate | None:
        """
        Raster indices in (row, col) format with origin at upper left of the grid.
        """
        return self._rowcol

    @property
    def xy(self) -> FloatCoordinate | None:
        """
        Geographic/projected (x, y) coordinates of the cell center in the CRS.
        """
        return self._xy

    def step(self):
        pass


class RasterLayer(RasterBase):
    """
    Some methods in `RasterLayer` are copied from `mesa.space.Grid`, including:

    __getitem__
    __iter__
    coord_iter
    iter_neighborhood
    get_neighborhood
    iter_neighbors
    get_neighbors  # copied and renamed to `get_neighboring_cells`
    out_of_bounds  # copied into `RasterBase`
    iter_cell_list_contents
    get_cell_list_contents

    Methods from `mesa.space.Grid` that are not copied over:

    torus_adj
    neighbor_iter
    move_agent
    place_agent
    _place_agent
    remove_agent
    is_cell_empty
    move_to_empty
    find_empty
    exists_empty_cells

    Another difference is that `mesa.space.Grid` has `self.grid: List[List[Agent | None]]`,
    whereas it is `self.cells: List[List[Cell]]` here in `RasterLayer`.
    """

    cells: list[list[Cell]]
    _neighborhood_cache: dict[Any, list[Coordinate]]
    _attributes: set[str]

    def __init__(
        self, width, height, crs, total_bounds, model, cell_cls: type[Cell] = Cell
    ):
        super().__init__(width, height, crs, total_bounds)
        self.model = model
        self.cell_cls = cell_cls
        self._initialize_cells()
        self._attributes = set()
        self._neighborhood_cache = {}

    def _update_transform(self) -> None:
        super()._update_transform()
        if getattr(self, "cells", None):
            self._sync_cell_xy()

    def _sync_cell_xy(self) -> None:
        for column in self.cells:
            for cell in column:
                row, col = cell.rowcol
                cell._xy = rio.transform.xy(self.transform, row, col, offset="center")

    def _initialize_cells(self) -> None:
        try:
            init_params = inspect.signature(self.cell_cls.__init__).parameters
        except (TypeError, ValueError):
            supports_legacy_pos_indices = False
        else:
            supports_legacy_pos_indices = (
                "pos" in init_params and "indices" in init_params
            )

        if supports_legacy_pos_indices:

            def make_cell(grid_x: int, grid_y: int, row_idx: int, col_idx: int, xy):
                # Backward-compatible path for legacy signature:
                # __init__(self, model, pos=None, indices=None, ...)
                cell = self.cell_cls(
                    self.model,
                    pos=(grid_x, grid_y),
                    indices=(row_idx, col_idx),
                )
                # Legacy constructor path does not accept xy; set it manually.
                cell._xy = xy
                return cell
        else:
            # New constructor path: __init__(self, model, pos=None, rowcol=None, xy=None, ...)
            # or: __init__(self, model, **kwargs)
            def make_cell(grid_x: int, grid_y: int, row_idx: int, col_idx: int, xy):
                return self.cell_cls(
                    self.model,
                    pos=(grid_x, grid_y),
                    rowcol=(row_idx, col_idx),
                    xy=xy,
                )

        self.cells = []
        for grid_x in range(self.width):
            col: list[Cell] = []
            for grid_y in range(self.height):
                row_idx, col_idx = self.height - grid_y - 1, grid_x
                xy = rio.transform.xy(self.transform, row_idx, col_idx, offset="center")
                cell = make_cell(grid_x, grid_y, row_idx, col_idx, xy)
                col.append(cell)
            self.cells.append(col)

    @property
    def attributes(self) -> set[str]:
        """
        Return the attributes of the cells in the raster layer.

        :return: Attributes of the cells in the raster layer.
        :rtype: Set[str]
        """
        return self._attributes

    @overload
    def __getitem__(self, index: int) -> list[Cell]: ...

    @overload
    def __getitem__(
        self, index: tuple[int | slice, int | slice]
    ) -> Cell | list[Cell]: ...

    @overload
    def __getitem__(self, index: Sequence[Coordinate]) -> list[Cell]: ...

    def __getitem__(
        self, index: int | Sequence[Coordinate] | tuple[int | slice, int | slice]
    ) -> Cell | list[Cell]:
        """
        Access contents from the grid.
        """

        if isinstance(index, int):
            # cells[x]
            return self.cells[index]

        if isinstance(index[0], tuple):
            # cells[(x1, y1), (x2, y2)]
            index = cast(Sequence[Coordinate], index)

            cells = []
            for pos in index:
                x1, y1 = pos
                cells.append(self.cells[x1][y1])
            return cells

        x, y = index

        if isinstance(x, int) and isinstance(y, int):
            # cells[x, y]
            x, y = cast(Coordinate, index)
            return self.cells[x][y]

        if isinstance(x, int):
            # cells[x, :]
            x = slice(x, x + 1)

        if isinstance(y, int):
            # grid[:, y]
            y = slice(y, y + 1)

        # cells[:, :]
        x, y = (cast(slice, x), cast(slice, y))
        cells = []
        for rows in self.cells[x]:
            for cell in rows[y]:
                cells.append(cell)
        return cells

    def __iter__(self) -> Iterator[Cell]:
        """
        Create an iterator that chains the rows of the cells together
        as if it is one list
        """

        return itertools.chain(*self.cells)

    def coord_iter(self) -> Iterator[tuple[Cell, int, int]]:
        """
        An iterator that returns coordinates as well as cell contents.
        """

        for row in range(self.width):
            for col in range(self.height):
                yield self.cells[row][col], row, col  # cell, x, y

    def apply_raster(
        self, data: np.ndarray, attr_name: str | Sequence[str] | None = None
    ) -> None:
        """
        Apply raster data to the cells.

        :param np.ndarray data: 3D numpy array with shape (bands, height, width).
        :param str | Sequence[str] | None attr_name: Attribute name(s) to be added to the
            cells. For multi-band rasters, pass a list of names with length equal to
            the number of bands, or a single base name to be suffixed per band. If None,
            names are generated. Default is None.
        :raises ValueError: If the shape of the data does not match the raster.
        """

        if data.ndim != 3 or data.shape[1:] != (self.height, self.width):
            raise ValueError(
                f"Data shape does not match raster shape. "
                f"Expected (*, {self.height}, {self.width}), received {data.shape}."
            )
        num_bands = data.shape[0]

        if num_bands == 1:
            if isinstance(attr_name, Sequence) and not isinstance(attr_name, str):
                if len(attr_name) != 1:
                    raise ValueError(
                        "attr_name sequence length must match the number of raster bands; "
                        f"expected {num_bands} band names, got {len(attr_name)}."
                    )
                names = [attr_name[0]]
            else:
                names = [cast(str | None, attr_name)]
        else:
            if isinstance(attr_name, Sequence) and not isinstance(attr_name, str):
                if len(attr_name) != num_bands:
                    raise ValueError(
                        "attr_name sequence length must match the number of raster bands; "
                        f"expected {num_bands} band names, got {len(attr_name)}."
                    )
                names = list(attr_name)
            elif isinstance(attr_name, str):
                names = [f"{attr_name}_{band_idx + 1}" for band_idx in range(num_bands)]
            else:
                names = [None] * num_bands

        def _default_attr_name() -> str:
            base = f"attribute_{len(self.cell_cls.__dict__)}"
            if base not in self._attributes:
                return base
            suffix = 1
            candidate = f"{base}_{suffix}"
            while candidate in self._attributes:
                suffix += 1
                candidate = f"{base}_{suffix}"
            return candidate

        for band_idx, name in enumerate(names):
            attr = _default_attr_name() if name is None else name
            self._attributes.add(attr)
            for grid_x in range(self.width):
                for grid_y in range(self.height):
                    setattr(
                        self.cells[grid_x][grid_y],
                        attr,
                        data[band_idx, self.height - grid_y - 1, grid_x],
                    )

    def get_raster(self, attr_name: str | Sequence[str] | None = None) -> np.ndarray:
        """
        Return the values of given attribute.

        :param str | Sequence[str] | None attr_name: Name(s) of attributes to be returned.
            If None, returns all attributes. Default is None.
        :return: The values of given attribute(s) as a numpy array with shape
            (bands, height, width).
        :rtype: np.ndarray
        """

        if isinstance(attr_name, str) and attr_name not in self.attributes:
            raise ValueError(
                f"Attribute {attr_name} does not exist. "
                f"Choose from {self.attributes}, or set `attr_name` to `None` to retrieve all."
            )
        if isinstance(attr_name, Sequence) and not isinstance(attr_name, str):
            missing = [name for name in attr_name if name not in self.attributes]
            if missing:
                raise ValueError(
                    f"Attribute {missing[0]} does not exist. "
                    f"Choose from {self.attributes}, or set `attr_name` to `None` to retrieve all."
                )
        if attr_name is None:
            num_bands = len(self.attributes)
            attr_names = self.attributes
        elif isinstance(attr_name, Sequence) and not isinstance(attr_name, str):
            num_bands = len(attr_name)
            attr_names = list(attr_name)
        else:
            num_bands = 1
            attr_names = [attr_name]
        data = np.empty((num_bands, self.height, self.width))
        for ind, name in enumerate(attr_names):
            for grid_x in range(self.width):
                for grid_y in range(self.height):
                    data[ind, self.height - grid_y - 1, grid_x] = getattr(
                        self.cells[grid_x][grid_y], name
                    )
        return data

    def iter_neighborhood(
        self,
        pos: Coordinate,
        moore: bool,
        include_center: bool = False,
        radius: int = 1,
    ) -> Iterator[Coordinate]:
        """
        Return an iterator over cell coordinates that are in the
        neighborhood of a certain point.

        :param Coordinate pos: Grid coordinate tuple (grid_x, grid_y) for the
            neighborhood to get. Origin is at lower left corner of the grid.
        :param bool moore: Whether to use Moore neighborhood or not. If True,
            return Moore neighborhood (including diagonals). If False, return
            Von Neumann neighborhood (exclude diagonals).
        :param bool include_center: If True, return the (grid_x, grid_y) cell as
            well. Otherwise, return surrounding cells only. Default is False.
        :param int radius: Radius, in cells, of the neighborhood. Default is 1.
        :return: An iterator over cell coordinates that are in the neighborhood.
            For example with radius 1, it will return list with number of elements
            equals at most 9 (8) if Moore, 5 (4) if Von Neumann (if not including
            the center).
        :rtype: Iterator[Coordinate]
        """

        yield from self.get_neighborhood(pos, moore, include_center, radius)

    def iter_neighbors(
        self,
        pos: Coordinate,
        moore: bool,
        include_center: bool = False,
        radius: int = 1,
    ) -> Iterator[Cell]:
        """
        Return an iterator over neighbors to a certain point.

        :param Coordinate pos: Grid coordinate tuple (grid_x, grid_y) for the
            neighborhood to get. Origin is at lower left corner of the grid.
        :param bool moore: Whether to use Moore neighborhood or not. If True,
            return Moore neighborhood (including diagonals). If False, return
            Von Neumann neighborhood (exclude diagonals).
        :param bool include_center: If True, return the (grid_x, grid_y) cell
            as well. Otherwise, return surrounding cells only. Default is False.
        :param int radius: Radius, in cells, of the neighborhood. Default is 1.
        :return: An iterator of cells that are in the neighborhood; at most 9 (8)
            if Moore, 5 (4) if Von Neumann (if not including the center).
        :rtype: Iterator[Cell]
        """

        neighborhood = self.get_neighborhood(pos, moore, include_center, radius)
        return self.iter_cell_list_contents(neighborhood)

    @accept_tuple_argument
    def iter_cell_list_contents(
        self, cell_list: Iterable[Coordinate]
    ) -> Iterator[Cell]:
        """
        Returns an iterator of the contents of the cells
        identified in cell_list.

        :param Iterable[Coordinate] cell_list: Array-like of grid (grid_x, grid_y) tuples,
            or single tuple (grid_x, grid_y). Origin is at lower left corner of the grid.
        :return: An iterator of the contents of the cells identified in cell_list.
        :rtype: Iterator[Cell]
        """

        # Note: filter(None, iterator) filters away an element of iterator that
        # is falsy. Hence, iter_cell_list_contents returns only non-empty
        # contents.
        return filter(None, (self.cells[x][y] for x, y in cell_list))

    @accept_tuple_argument
    def get_cell_list_contents(self, cell_list: Iterable[Coordinate]) -> list[Cell]:
        """
        Returns a list of the contents of the cells
        identified in cell_list.

        Note: this method returns a list of cells.

        :param Iterable[Coordinate] cell_list: Array-like of grid (grid_x, grid_y) tuples,
            or single tuple (grid_x, grid_y). Origin is at lower left corner of the grid.
        :return: A list of the contents of the cells identified in cell_list.
        :rtype: List[Cell]
        """

        return list(self.iter_cell_list_contents(cell_list))

    def get_neighborhood(
        self,
        pos: Coordinate,
        moore: bool,
        include_center: bool = False,
        radius: int = 1,
    ) -> list[Coordinate]:
        """
        Return a list of cell coordinates that are in the
        neighborhood of a certain point.

        :param Coordinate pos: Grid coordinate tuple (grid_x, grid_y) for the
            neighborhood to get. Origin is at lower left corner of the grid.
        :param bool moore: Whether to use Moore neighborhood or not. If True,
            return Moore neighborhood (including diagonals). If False, return
            Von Neumann neighborhood (exclude diagonals).
        :param bool include_center: If True, return the (grid_x, grid_y) cell as
            well. Otherwise, return surrounding cells only. Default is False.
        :param int radius: Radius, in cells, of the neighborhood. Default is 1.
        :return: A list of cell coordinates that are in the neighborhood.
            For example with radius 1, it will return list with number of elements
            equals at most 9 (8) if Moore, 5 (4) if Von Neumann (if not including
            the center).
        :rtype: List[Coordinate]
        """
        cache_key = (pos, moore, include_center, radius)
        neighborhood = self._neighborhood_cache.get(cache_key, None)

        if neighborhood is None:
            coordinates: set[Coordinate] = set()

            x, y = pos
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    if dx == 0 and dy == 0 and not include_center:
                        continue
                    # Skip coordinates that are outside manhattan distance
                    if not moore and abs(dx) + abs(dy) > radius:
                        continue

                    coord = (x + dx, y + dy)

                    if self.out_of_bounds(coord):
                        continue
                    coordinates.add(coord)

            neighborhood = sorted(coordinates)
            self._neighborhood_cache[cache_key] = neighborhood

        return neighborhood

    def get_neighboring_cells(
        self,
        pos: Coordinate,
        moore: bool,
        include_center: bool = False,
        radius: int = 1,
    ) -> list[Cell]:
        neighboring_cell_idx = self.get_neighborhood(pos, moore, include_center, radius)
        return [self.cells[idx[0]][idx[1]] for idx in neighboring_cell_idx]

    def to_crs(self, crs, inplace=False) -> RasterLayer | None:
        """
        Transform the raster layer to a new coordinate reference system.

        :param crs: The coordinate reference system to transform to.
        :param inplace: Whether to transform the raster layer in place or
            return a new raster layer. Defaults to False.
        :return: The transformed raster layer if not inplace.
        :rtype: RasterLayer | None
        """

        super()._to_crs_check(crs)
        layer = self if inplace else copy.deepcopy(self)

        src_crs = rio.crs.CRS.from_user_input(layer.crs)
        dst_crs = rio.crs.CRS.from_user_input(crs)
        if not layer.crs.is_exact_same(crs):
            transform, _, _ = calculate_default_transform(
                src_crs,
                dst_crs,
                self.width,
                self.height,
                *layer.total_bounds,
            )
            layer._total_bounds = [
                *transform_bounds(src_crs, dst_crs, *layer.total_bounds)
            ]
            layer.crs = crs
            layer._transform = transform
            if getattr(layer, "cells", None):
                layer._sync_cell_xy()

        if not inplace:
            return layer

    def to_image(self, colormap) -> ImageLayer:
        """
        Returns an ImageLayer colored by the provided colormap.
        """

        values = np.empty(shape=(4, self.height, self.width))
        for cell in self:
            row, col = cell.rowcol
            values[:, row, col] = colormap(cell)
        return ImageLayer(values=values, crs=self.crs, total_bounds=self.total_bounds)

    @classmethod
    def from_file(
        cls,
        raster_file: str,
        model: Model,
        cell_cls: type[Cell] = Cell,
        attr_name: str | Sequence[str] | None = None,
        rio_opener: Callable | None = None,
    ) -> RasterLayer:
        """
        Creates a RasterLayer from a raster file.

        :param str raster_file: Path to the raster file.
        :param Type[Cell] cell_cls: The class of the cells in the layer.
        :param str | Sequence[str] | None attr_name: Attribute name(s) to use for the cell
            values. For multi-band rasters, pass a list of names with length equal to
            the number of bands, or a single base name to be suffixed per band. If None,
            names are generated. Default is None.
        :param Callable | None rio_opener: A callable passed to Rasterio open() function.
        """

        with rio.open(raster_file, "r", opener=rio_opener) as dataset:
            values = dataset.read()
            _, height, width = values.shape
            total_bounds = [
                dataset.bounds.left,
                dataset.bounds.bottom,
                dataset.bounds.right,
                dataset.bounds.top,
            ]
            obj = cls(width, height, dataset.crs, total_bounds, model, cell_cls)
            obj._transform = dataset.transform
            obj._sync_cell_xy()
            obj.apply_raster(values, attr_name=attr_name)
            return obj

    def to_file(
        self,
        raster_file: str,
        attr_name: str | Sequence[str] | None = None,
        driver: str = "GTiff",
    ) -> None:
        """
        Writes a raster layer to a file.

        :param str raster_file: The path to the raster file to write to.
        :param str | Sequence[str] | None attr_name: The name(s) of attributes to write
            to the raster. If None, all attributes are written. Default is None.
        :param str driver: The GDAL driver to use for writing the raster file.
            Default is 'GTiff'. See GDAL docs at https://gdal.org/drivers/raster/index.html.
        """

        data = self.get_raster(attr_name)
        with rio.open(
            raster_file,
            "w",
            driver=driver,
            width=self.width,
            height=self.height,
            count=data.shape[0],
            dtype=data.dtype,
            crs=self.crs,
            transform=self.transform,
        ) as dataset:
            dataset.write(data)


class ImageLayer(RasterBase):
    _values: np.ndarray

    def __init__(self, values, crs, total_bounds):
        """
        Initializes an ImageLayer.

        :param values: The values of the image layer.
        :param crs: The coordinate reference system of the image layer.
        :param total_bounds: The bounds of the image layer in [min_x, min_y, max_x, max_y] format.
        """

        super().__init__(
            width=values.shape[2],
            height=values.shape[1],
            crs=crs,
            total_bounds=total_bounds,
        )
        self._values = values.copy()

    @property
    def values(self) -> np.ndarray:
        """
        Returns the values of the image layer.

        :return: The values of the image layer.
        :rtype: np.ndarray
        """

        return self._values

    @values.setter
    def values(self, values: np.ndarray) -> None:
        """
        Sets the values of the image layer.

        :param np.ndarray values: The values of the image layer.
        """

        self._values = values
        self._width = values.shape[2]
        self._height = values.shape[1]
        self._update_transform()

    def to_crs(self, crs, inplace=False) -> ImageLayer | None:
        """
        Transform the image layer to a new coordinate reference system.

        :param crs: The coordinate reference system to transform to.
        :param inplace: Whether to transform the image layer in place or
            return a new image layer. Defaults to False.
        :return: The transformed image layer if not inplace.
        :rtype: ImageLayer | None
        """
        super()._to_crs_check(crs)
        layer = self if inplace else copy.copy(self)

        src_crs = rio.crs.CRS.from_user_input(layer.crs)
        dst_crs = rio.crs.CRS.from_user_input(crs)
        if not layer.crs.is_exact_same(crs):
            num_bands, src_height, src_width = self.values.shape
            transform, dst_width, dst_height = calculate_default_transform(
                src_crs,
                dst_crs,
                src_width,
                src_height,
                *layer.total_bounds,
            )
            dst = np.empty(shape=(num_bands, dst_height, dst_width))
            for i, band in enumerate(layer.values):
                reproject(
                    source=band,
                    destination=dst[i],
                    src_transform=layer.transform,
                    src_crs=src_crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest,
                )
            layer._total_bounds = [
                *transform_bounds(src_crs, dst_crs, *layer.total_bounds)
            ]
            layer._values = dst
            layer._height = layer._values.shape[1]
            layer._width = layer._values.shape[2]
            layer.crs = crs
            layer._transform = transform
        if not inplace:
            return layer

    @classmethod
    def from_file(cls, image_file) -> ImageLayer:
        """
        Creates an ImageLayer from an image file.

        :param image_file: The path to the image file.
        :return: The ImageLayer.
        :rtype: ImageLayer
        """

        with rio.open(image_file, "r") as dataset:
            values = dataset.read()
            total_bounds = [
                dataset.bounds.left,
                dataset.bounds.bottom,
                dataset.bounds.right,
                dataset.bounds.top,
            ]
            obj = cls(values=values, crs=dataset.crs, total_bounds=total_bounds)
            obj._transform = dataset.transform
            return obj

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(crs={self.crs}, total_bounds={self.total_bounds}, values={self.values!r})"
