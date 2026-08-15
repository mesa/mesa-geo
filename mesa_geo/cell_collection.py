"""
Raster Cell Collection
----------------------

A set-like container for selecting, querying, and mutating subsets of cells
in a :class:`~mesa_geo.raster_layers.RasterLayer` without materializing
individual :class:`~mesa_geo.raster_layers.Cell` objects.

Internally backed by a boolean mask in (row, col) space matching
``RasterLayer._data`` layout, so aggregation and bulk writes are single
NumPy operations.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from mesa_geo.raster_layers import Cell, RasterLayer


class _CellTracer:
    """
    A duck-typed tracer that intercepts attribute accesses to enable
    vectorized NumPy operations on bands without instantiating Cells.
    """

    __slots__ = ("_layer",)

    def __init__(self, layer: RasterLayer) -> None:
        self._layer = layer

    def __getattr__(self, name: str) -> np.ndarray:
        if name in self._layer._data:
            return self._layer._data[name]
        # Propagate AttributeError for non-band attributes
        raise AttributeError(f"Band '{name}' not found on _CellTracer.")


class RasterCellCollection:
    """
    A mask-backed collection of cells from a single
    :class:`~mesa_geo.raster_layers.RasterLayer`.

    The collection stores a boolean mask of shape ``(height, width)`` in
    **(row, col) space** — the same layout as ``RasterLayer._data`` — so
    set operations, aggregation, and bulk reads/writes are O(1) NumPy
    broadcasts rather than per-cell Python loops.

    Cells are materialized lazily only when iterated.  Methods such as
    :meth:`get`, :meth:`set`, :meth:`agg`, and :meth:`count` never
    construct a :class:`~mesa_geo.raster_layers.Cell`.

    :param layer: The parent :class:`~mesa_geo.raster_layers.RasterLayer`.
    :param mask: Optional boolean mask of shape ``(height, width)``.
        ``None`` (default) selects all cells.
    """

    __slots__ = ("_layer", "_mask")

    def __init__(self, layer: RasterLayer, mask: np.ndarray | None = None) -> None:
        """
        Initialize a RasterCellCollection.

        :param layer: The parent :class:`~mesa_geo.raster_layers.RasterLayer`
            this collection belongs to.
        :param mask: A boolean ``np.ndarray`` of shape
            ``(layer.height, layer.width)``.  If ``None``, all cells are
            selected.
        :raises ValueError: If *mask* is not ``None`` and its shape does not
            match ``(layer.height, layer.width)``.
        """
        self._layer: RasterLayer = layer

        if mask is None:
            self._mask: np.ndarray = np.ones((layer.height, layer.width), dtype=bool)
        else:
            if mask.shape != (layer.height, layer.width):
                raise ValueError(
                    f"Mask shape {mask.shape} does not match layer shape "
                    f"({layer.height}, {layer.width})."
                )
            self._mask = np.array(mask, dtype=bool, copy=True)

    # ------------------------------------------------------------------
    # Single conversion point — (row, col) → Cell
    # ------------------------------------------------------------------

    def _to_cells(self):
        """
        Yield :class:`~mesa_geo.raster_layers.Cell` objects for every
        ``True`` entry in the mask.

        This is the **single** place where the vertical flip between
        (row, col) raster space and (grid_x, grid_y) cell-grid space is
        performed.  ``np.nonzero`` yields in row-major (top-to-bottom,
        left-to-right) order, which differs from
        ``RasterLayer.__iter__``'s column-major order.

        :return: Generator of :class:`~mesa_geo.raster_layers.Cell`.
        :rtype: Iterator[Cell]
        """
        rows, cols = np.nonzero(self._mask)
        h = self._layer.height
        for r, c in zip(rows, cols):
            yield self._layer.cells[c][h - r - 1]

    # ------------------------------------------------------------------
    # Container protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """
        Return the number of selected cells.

        Does **not** materialize any :class:`~mesa_geo.raster_layers.Cell`
        objects.

        :return: Count of ``True`` entries in the mask.
        :rtype: int
        """
        return int(self._mask.sum())

    def __iter__(self):
        """
        Iterate over the selected cells, yielding
        :class:`~mesa_geo.raster_layers.Cell` instances.

        Cells are yielded in ``np.nonzero`` order (row-major,
        top-to-bottom then left-to-right), which differs from
        ``RasterLayer.__iter__``'s column-major order.

        :return: Iterator of :class:`~mesa_geo.raster_layers.Cell`.
        :rtype: Iterator[Cell]
        """
        return self._to_cells()

    def __contains__(self, cell: Cell) -> bool:
        """
        Check whether *cell* is in this collection by testing its
        ``rowcol`` against the mask.

        :param cell: A :class:`~mesa_geo.raster_layers.Cell`.
        :return: ``True`` if the cell's position is selected.
        :rtype: bool

        .. note::
            This method does **not** verify that *cell* belongs to the
            same parent layer as this collection.  The ``_layer``
            attribute is being redesigned upstream, so a cross-layer
            identity check is deferred.
        """
        rowcol = cell.rowcol
        if rowcol is None:
            return False
        row, col = rowcol
        if row < 0 or row >= self._layer.height or col < 0 or col >= self._layer.width:
            return False
        return bool(self._mask[row, col])

    def to_list(self) -> list[Cell]:
        """
        Return all selected cells as a list.

        :return: List of :class:`~mesa_geo.raster_layers.Cell` in
            ``np.nonzero`` (row-major) order.
        :rtype: list[Cell]
        """
        return list(self)

    # ------------------------------------------------------------------
    # Set operations
    # ------------------------------------------------------------------

    def __and__(self, other: RasterCellCollection) -> RasterCellCollection:
        """
        Return the intersection of two collections (mask AND).

        :param other: Another :class:`RasterCellCollection` from the
            **same** parent layer.
        :return: New :class:`RasterCellCollection` with the intersected
            mask.
        :rtype: RasterCellCollection
        :raises ValueError: If *other* does not share the same parent
            layer.
        """
        if self._layer is not other._layer:
            raise ValueError("Cannot combine collections from different layers.")
        return RasterCellCollection(self._layer, self._mask & other._mask)

    def __or__(self, other: RasterCellCollection) -> RasterCellCollection:
        """
        Return the union of two collections (mask OR).

        :param other: Another :class:`RasterCellCollection` from the
            **same** parent layer.
        :return: New :class:`RasterCellCollection` with the unioned mask.
        :rtype: RasterCellCollection
        :raises ValueError: If *other* does not share the same parent
            layer.
        """
        if self._layer is not other._layer:
            raise ValueError("Cannot combine collections from different layers.")
        return RasterCellCollection(self._layer, self._mask | other._mask)

    def __invert__(self) -> RasterCellCollection:
        """
        Return the complement of this collection (mask NOT).

        :return: New :class:`RasterCellCollection` with the inverted mask.
        :rtype: RasterCellCollection
        """
        return RasterCellCollection(self._layer, ~self._mask)

    # ------------------------------------------------------------------
    # Selection and Filtering
    # ------------------------------------------------------------------

    def select(
        self,
        filter_func: Callable[[Cell], bool | np.ndarray] | None = None,
        at_most: int | float = float("inf"),
        inplace: bool = False,
    ) -> RasterCellCollection:
        """
        Select a subset of cells based on a filter function and/or quantity limit.

        If *filter_func* is provided, it is first evaluated with a
        :class:`_CellTracer` proxy to attempt single-operation NumPy
        vectorization. If this fails due to a tracing error (e.g. non-vectorizable
        operations), it falls back to a per-cell Python loop. However, if the
        evaluation succeeds but returns an invalid result (e.g. a scalar or non-boolean
        array), a :exc:`ValueError` is raised instead.

        .. note::
            For optimal performance, write predicates using bitwise operators
            (``&``, ``|``, ``~``) with parentheses, which can be fully vectorized
            (e.g. ``lambda c: (c.elevation > 100) & (c.population < 25)``).
            Operations that fall back to a per-cell loop will execute the predicate
            twice (once during the failed tracing attempt, and again per cell). Ensure
            your predicate does not have unintended side effects.

        :param filter_func: A function that takes a cell (or tracer) and
            returns a boolean or boolean array. Default is ``None``.
        :param at_most: The maximum amount of cells to select. Defaults to
            infinity.
            - If an integer, at most the first ``n`` matching cells are selected.
            - If a float between 0 and 1 (inclusive), at most that fraction of
              the original cells in this collection are selected.
            - Booleans are rejected with :class:`TypeError`.
        :param inplace: If ``True``, modifies the current collection;
            otherwise, returns a new collection. Defaults to ``False``.
        :return: A RasterCellCollection containing the selected cells.
        :rtype: RasterCellCollection
        """
        new_mask = self._mask.copy()

        if filter_func is not None:
            tracer = _CellTracer(self._layer)
            needs_fallback = False
            result = None
            try:
                result = filter_func(tracer)
            except AttributeError:
                # Missing band accessed — propagate error
                raise
            except Exception:
                # Tracer evaluation itself failed (e.g. math.sqrt on
                # array, Python `and` on arrays, chained comparison) —
                # genuinely non-vectorizable, fall through to loop.
                needs_fallback = True

            if not needs_fallback:
                # The tracer successfully evaluated the predicate.
                # Now validate the result shape and dtype.
                if (
                    isinstance(result, np.ndarray)
                    and result.dtype == bool
                    and result.shape == new_mask.shape
                ):
                    new_mask &= result
                elif isinstance(result, np.ndarray):
                    # Tracer produced an array but wrong dtype or shape —
                    # this is a bad predicate, not a non-vectorizable one.
                    raise ValueError(
                        f"filter_func must return a boolean array of shape "
                        f"{new_mask.shape}, got dtype={result.dtype}, "
                        f"shape={result.shape}."
                    )
                else:
                    # Tracer returned a scalar (e.g. .sum() > 100) —
                    # bad predicate, not non-vectorizable.
                    raise ValueError(
                        f"filter_func must return a boolean array, "
                        f"got scalar {type(result).__name__}."
                    )
            else:
                # Per-cell fallback — errors from the predicate itself
                # (e.g. calling .sum() on a float) are NOT caught here;
                # they propagate immediately to the caller.
                fallback_mask = np.zeros_like(new_mask)
                for cell in self:
                    if filter_func(cell):
                        r, c = cell.rowcol  # type: ignore[misc]
                        fallback_mask[r, c] = True
                new_mask &= fallback_mask

        if isinstance(at_most, bool):
            raise TypeError(f"at_most must be int or float, got bool ({at_most!r}).")
        if at_most < float("inf"):
            if isinstance(at_most, float) and 0.0 < at_most <= 1.0:
                limit = int(at_most * len(self))
            else:
                limit = int(at_most)

            true_indices = np.nonzero(new_mask)
            if len(true_indices[0]) > limit:
                # Set all elements past `limit` to False
                rows = true_indices[0][limit:]
                cols = true_indices[1][limit:]
                new_mask[rows, cols] = False

        if inplace:
            self._mask = new_mask
            return self
        else:
            return RasterCellCollection(self._layer, new_mask)

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """
        Return a human-readable representation.

        :return: String showing layer dimensions and selected cell count.
        :rtype: str
        """
        h, w = self._layer.height, self._layer.width
        n = len(self)
        return f"RasterCellCollection({w}x{h} layer, {n}/{h * w} cells selected)"

    # ------------------------------------------------------------------
    # Data Access and Mutation
    # ------------------------------------------------------------------

    def get(
        self,
        attr_names: str | list[str],
        handle_missing: str = "error",
        default_value: Any = None,
    ) -> np.ndarray | list[np.ndarray]:
        """
        Get the values of one or more attributes for the selected cells.

        Returns a 1D NumPy array for each requested attribute, containing the
        values of the selected cells in mask order (row-major). Note that
        unlike :meth:`~mesa.agentset.AbstractAgentSet.get`, which returns
        lists, this method returns NumPy arrays which are more useful for
        raster data.

        .. note::
            Because this method uses NumPy boolean indexing on the underlying
            band arrays, the returned arrays are always copies, not views.

        :param attr_names: A string or list of strings naming the bands to get.
        :param handle_missing: How to handle missing bands. ``"error"`` raises
            an :class:`AttributeError`; ``"default"`` returns *default_value*.
        :param default_value: Value to return if the band is missing and
            *handle_missing* is ``"default"``.
        :return: A 1D NumPy array if *attr_names* is a string, or a list of
            NumPy arrays if *attr_names* is a list.
        :raises AttributeError: If *handle_missing* is ``"error"`` and a band
            does not exist.
        :raises ValueError: If *handle_missing* is invalid.
        """
        if handle_missing not in ("error", "default"):
            raise ValueError(
                f"handle_missing must be 'error' or 'default', got {handle_missing!r}"
            )

        is_single = isinstance(attr_names, str)
        names = [attr_names] if is_single else attr_names

        results = []
        for name in names:
            if name in self._layer._data:
                results.append(self._layer._data[name][self._mask])
            elif handle_missing == "error":
                raise AttributeError(f"Band '{name}' not found on the layer.")
            elif handle_missing == "default":
                results.append(np.full(len(self), default_value))

        if is_single:
            return results[0]
        return results

    def set(self, attr_name: str, value: Any) -> RasterCellCollection:
        """
        Set the values of a given attribute for the selected cells.

        :param attr_name: The name of the band to modify.
        :param value: The value to set. This can be:
            - A scalar value.
            - A 1D array of the same length as the number of selected cells.
            - A callable that takes a 1D array of the current values and
              returns a new array of values.
        :return: This collection itself, for chaining.
        :rtype: RasterCellCollection
        :raises AttributeError: If the band does not exist.
        """
        if attr_name not in self._layer._data:
            raise AttributeError(f"Band '{attr_name}' not found on the layer.")

        if callable(value):
            current_vals = self._layer._data[attr_name][self._mask]
            self._layer._data[attr_name][self._mask] = value(current_vals)
        else:
            self._layer._data[attr_name][self._mask] = value

        return self

    def agg(
        self,
        attribute: str,
        func: Callable | list[Callable] | tuple[Callable, ...],
    ) -> Any:
        """
        Aggregate the values of an attribute over the selected cells.

        :param attribute: The name of the band to aggregate.
        :param func: A callable (e.g. ``np.mean``) or an iterable of callables
            (e.g. ``[np.min, np.max]``).
        :return: A single value if *func* is a callable, or a tuple of values
            if *func* is an iterable of callables.
        :raises ValueError: If the collection is empty.
        :raises AttributeError: If the band does not exist.
        """
        if len(self) == 0:
            raise ValueError("Cannot aggregate an empty RasterCellCollection.")

        if attribute not in self._layer._data:
            raise AttributeError(f"Band '{attribute}' not found on the layer.")

        vals = self._layer._data[attribute][self._mask]

        if isinstance(func, (list, tuple)):
            return tuple(f(vals) for f in func)
        return func(vals)

    def count(self) -> int:
        """
        Return the number of selected cells. Alias for ``len(self)``.

        :return: Count of selected cells.
        :rtype: int
        """
        return len(self)
