"""Tests for RasterCellCollection - container basics (Steps 1-2)."""

import math
import unittest

import mesa
import numpy as np

import mesa_geo as mg
from mesa_geo.cell_collection import RasterCellCollection


class TestRasterCellCollectionContainer(unittest.TestCase):
    """Container-protocol and set-operation tests."""

    def setUp(self) -> None:
        self.model = mesa.Model()
        self.layer = mg.RasterLayer(
            width=3,
            height=4,
            crs="epsg:4326",
            total_bounds=[0, 0, 3, 4],
            model=self.model,
        )
        # Add a band so cells have data to read
        self.elevation = np.arange(12, dtype=float).reshape(4, 3)
        self.layer.set_band("elevation", self.elevation)

    # ------------------------------------------------------------------
    # __init__
    # ------------------------------------------------------------------

    def test_default_mask_selects_all(self):
        """mask=None should select every cell."""
        coll = RasterCellCollection(self.layer)
        self.assertEqual(len(coll), self.layer.height * self.layer.width)

    def test_custom_mask(self):
        """A custom mask should be respected."""
        mask = np.zeros((4, 3), dtype=bool)
        mask[0, 0] = True
        mask[2, 1] = True
        coll = RasterCellCollection(self.layer, mask)
        self.assertEqual(len(coll), 2)

    def test_mask_shape_mismatch_raises(self):
        """A mask with wrong shape should raise ValueError."""
        bad_mask = np.ones((10, 10), dtype=bool)
        with self.assertRaises(ValueError):
            RasterCellCollection(self.layer, bad_mask)

    # ------------------------------------------------------------------
    # __len__
    # ------------------------------------------------------------------

    def test_len_all(self):
        coll = RasterCellCollection(self.layer)
        self.assertEqual(len(coll), 12)

    def test_len_empty(self):
        mask = np.zeros((4, 3), dtype=bool)
        coll = RasterCellCollection(self.layer, mask)
        self.assertEqual(len(coll), 0)

    def test_len_partial(self):
        mask = np.zeros((4, 3), dtype=bool)
        mask[1, :] = True  # one full row → 3 cells
        coll = RasterCellCollection(self.layer, mask)
        self.assertEqual(len(coll), 3)

    # ------------------------------------------------------------------
    # __iter__ and _to_cells
    # ------------------------------------------------------------------

    def test_iter_yields_cells(self):
        """Iterating should yield Cell objects."""
        coll = RasterCellCollection(self.layer)
        cells = list(coll)
        self.assertEqual(len(cells), 12)
        for cell in cells:
            self.assertIsInstance(cell, mg.Cell)

    def test_iter_single_cell(self):
        """Selecting a single cell at a known (row, col) should yield the
        correct Cell with the correct rowcol."""
        mask = np.zeros((4, 3), dtype=bool)
        mask[0, 2] = True  # row=0, col=2
        coll = RasterCellCollection(self.layer, mask)
        cells = list(coll)
        self.assertEqual(len(cells), 1)
        cell = cells[0]
        self.assertEqual(cell.rowcol, (0, 2))

    # ------------------------------------------------------------------
    # Orientation test (Step 0 point 3 / Step 6 item 2)
    # ------------------------------------------------------------------

    def test_orientation_top_row_maps_to_max_grid_y(self):
        """Selecting the top row of _data (row=0) must yield cells with
        grid_y == height - 1, NOT grid_y == 0.  This is the vertical-flip
        correctness test."""
        mask = np.zeros((4, 3), dtype=bool)
        mask[0, :] = True  # top row in raster space
        coll = RasterCellCollection(self.layer, mask)
        cells = list(coll)
        self.assertEqual(len(cells), 3)
        for col, cell in enumerate(cells):
            self.assertIs(
                cell,
                self.layer.cells[col][self.layer.height - 1],
                f"Cell at rowcol={cell.rowcol} should be "
                f"cells[{col}][{self.layer.height - 1}]",
            )

    # ------------------------------------------------------------------
    # __contains__
    # ------------------------------------------------------------------

    def test_contains_selected_cell(self):
        mask = np.zeros((4, 3), dtype=bool)
        mask[1, 2] = True
        coll = RasterCellCollection(self.layer, mask)
        # The cell at grid_x=2, grid_y=height-1-1=2 → rowcol=(1, 2)
        cell = self.layer.cells[2][2]  # cells[col_idx][grid_y]
        self.assertEqual(cell.rowcol, (1, 2))
        self.assertIn(cell, coll)

    def test_not_contains_unselected_cell(self):
        mask = np.zeros((4, 3), dtype=bool)
        mask[1, 2] = True
        coll = RasterCellCollection(self.layer, mask)
        # A cell NOT in the mask
        cell = self.layer.cells[0][0]
        self.assertNotIn(cell, coll)

    # ------------------------------------------------------------------
    # to_list
    # ------------------------------------------------------------------

    def test_to_list_matches_iter(self):
        mask = np.zeros((4, 3), dtype=bool)
        mask[0, :] = True
        mask[3, :] = True
        coll = RasterCellCollection(self.layer, mask)
        self.assertEqual(coll.to_list(), list(coll))

    # ------------------------------------------------------------------
    # Set operations: &, |, ~
    # ------------------------------------------------------------------

    def test_and(self):
        mask_a = np.zeros((4, 3), dtype=bool)
        mask_a[0, :] = True
        mask_a[1, :] = True
        mask_b = np.zeros((4, 3), dtype=bool)
        mask_b[1, :] = True
        mask_b[2, :] = True

        a = RasterCellCollection(self.layer, mask_a)
        b = RasterCellCollection(self.layer, mask_b)
        result = a & b
        # Intersection: only row 1
        self.assertEqual(len(result), 3)
        np.testing.assert_array_equal(result._mask, mask_a & mask_b)

    def test_or(self):
        mask_a = np.zeros((4, 3), dtype=bool)
        mask_a[0, :] = True
        mask_b = np.zeros((4, 3), dtype=bool)
        mask_b[1, :] = True

        a = RasterCellCollection(self.layer, mask_a)
        b = RasterCellCollection(self.layer, mask_b)
        result = a | b
        # Union: rows 0 and 1
        self.assertEqual(len(result), 6)
        np.testing.assert_array_equal(result._mask, mask_a | mask_b)

    def test_invert(self):
        mask = np.zeros((4, 3), dtype=bool)
        mask[0, 0] = True
        coll = RasterCellCollection(self.layer, mask)
        inv = ~coll
        self.assertEqual(len(inv), 12 - 1)
        np.testing.assert_array_equal(inv._mask, ~mask)

    def test_set_ops_different_layers_raises(self):
        """& and | on collections from different layers should raise."""
        layer2 = mg.RasterLayer(
            width=5,
            height=6,
            crs="epsg:4326",
            total_bounds=[0, 0, 5, 6],
            model=self.model,
        )
        a = RasterCellCollection(self.layer)
        b = RasterCellCollection(layer2)
        with self.assertRaises(ValueError):
            _ = a & b
        with self.assertRaises(ValueError):
            _ = a | b

    # ------------------------------------------------------------------
    # __repr__
    # ------------------------------------------------------------------

    def test_repr(self):
        coll = RasterCellCollection(self.layer)
        r = repr(coll)
        self.assertIn("3x4", r)  # width x height
        self.assertIn("12/12", r)

    def test_repr_partial(self):
        mask = np.zeros((4, 3), dtype=bool)
        mask[0, 0] = True
        coll = RasterCellCollection(self.layer, mask)
        r = repr(coll)
        self.assertIn("1/12", r)

    # ------------------------------------------------------------------
    # Agreement: collection iteration vs Cell proxy reads
    # ------------------------------------------------------------------

    def test_agreement_iter_vs_cell_proxy(self):
        """Every cell yielded by the collection should have an elevation
        value that matches a direct cell.elevation read."""
        coll = RasterCellCollection(self.layer)
        for cell in coll:
            row, col = cell.rowcol
            expected = self.elevation[row, col]
            self.assertEqual(
                cell.elevation,
                expected,
                f"Mismatch at rowcol={cell.rowcol}: "
                f"cell.elevation={cell.elevation}, expected={expected}",
            )

    def test_mask_mutation_after_construction(self):
        """Mutating the input mask after construction must not change the
        collection — the copy=True in __init__ should insulate it."""
        mask = np.zeros((4, 3), dtype=bool)
        mask[0, 0] = True
        coll = RasterCellCollection(self.layer, mask)
        self.assertEqual(len(coll), 1)
        # mutate the original mask
        mask[:] = True
        # collection should still have only 1 cell
        self.assertEqual(len(coll), 1)


class TestRasterCellCollectionSelect(unittest.TestCase):
    """Tests for the select() method and _CellTracer."""

    def setUp(self) -> None:
        self.model = mesa.Model()
        self.layer = mg.RasterLayer(
            width=3,
            height=4,
            crs="epsg:4326",
            total_bounds=[0, 0, 3, 4],
            model=self.model,
        )
        self.elevation = np.arange(12, dtype=float).reshape(4, 3)
        self.layer.set_band("elevation", self.elevation)

    def test_select_none(self):
        """filter_func=None should return a copy of the collection (or same cells)."""
        coll = RasterCellCollection(self.layer)
        res = coll.select()
        self.assertIsNot(coll, res)
        self.assertEqual(len(res), 12)

    def test_select_vectorized(self):
        """A simple mask-based operation should vectorize."""
        coll = RasterCellCollection(self.layer)
        # Cells with elevation > 5
        res = coll.select(lambda c: c.elevation > 5)
        self.assertEqual(len(res), 6)  # 6, 7, 8, 9, 10, 11
        for cell in res:
            self.assertGreater(cell.elevation, 5)

    def test_select_fallback(self):
        """An operation that is not vectorizable (e.g., math.sqrt or pos) should loop."""
        coll = RasterCellCollection(self.layer)

        def f(c):
            # This will raise TypeError when evaluated with an ndarray
            return math.sqrt(c.elevation) > 2.5

        res = coll.select(f)
        self.assertEqual(len(res), 5)  # 7, 8, 9, 10, 11
        for cell in res:
            self.assertGreater(math.sqrt(cell.elevation), 2.5)

    def test_select_attribute_error_propagates(self):
        """Accessing a missing band should raise AttributeError, not fallback."""
        coll = RasterCellCollection(self.layer)
        with self.assertRaises(AttributeError):
            coll.select(lambda c: c.missing_band > 5)

    def test_select_at_most_int(self):
        """at_most=<int> should return at most that many cells."""
        coll = RasterCellCollection(self.layer)
        res = coll.select(lambda c: c.elevation >= 0, at_most=5)
        self.assertEqual(len(res), 5)
        # Should be the first 5 in iteration order (row-major: elevation 0, 1, 2, 3, 4)
        for expected_elev, cell in zip(range(5), res):
            self.assertEqual(cell.elevation, expected_elev)

    def test_select_at_most_float(self):
        """at_most=<float> should return a fraction of the current collection size."""
        coll = RasterCellCollection(self.layer)
        # len(coll) is 12. 0.5 * 12 = 6
        res = coll.select(at_most=0.5)
        self.assertEqual(len(res), 6)

    def test_select_inplace(self):
        """inplace=True should mutate the collection."""
        coll = RasterCellCollection(self.layer)
        res = coll.select(lambda c: c.elevation > 5, inplace=True)
        self.assertIs(coll, res)
        self.assertEqual(len(coll), 6)

    def test_select_at_most_float_one_returns_all(self):
        """at_most=1.0 means 100% — should return the full selection."""
        coll = RasterCellCollection(self.layer)
        res = coll.select(at_most=1.0)
        self.assertEqual(len(res), 12)

    def test_select_at_most_bool_raises(self):
        """at_most=True (a bool) should raise TypeError."""
        coll = RasterCellCollection(self.layer)
        with self.assertRaises(TypeError):
            coll.select(at_most=True)
        with self.assertRaises(TypeError):
            coll.select(at_most=False)

    # ------------------------------------------------------------------
    # Fallback tests: non-vectorizable predicate forms
    # ------------------------------------------------------------------

    def test_select_fallback_python_and(self):
        """Python `and` in a lambda forces fallback; result must match
        the equivalent vectorized predicate."""
        coll = RasterCellCollection(self.layer)
        expected = coll.select(lambda c: (c.elevation > 5) & (c.elevation < 10))
        got = coll.select(lambda c: c.elevation > 5 and c.elevation < 10)
        np.testing.assert_array_equal(got._mask, expected._mask)

    def test_select_fallback_chained_comparison(self):
        """Chained comparison ``5 < x < 10`` forces fallback."""
        coll = RasterCellCollection(self.layer)
        expected = coll.select(lambda c: (c.elevation > 5) & (c.elevation < 10))
        got = coll.select(lambda c: 5 < c.elevation < 10)
        np.testing.assert_array_equal(got._mask, expected._mask)

    def test_select_fallback_if_in_lambda(self):
        """Conditional expression inside the lambda forces fallback."""
        coll = RasterCellCollection(self.layer)
        expected = coll.select(lambda c: c.elevation > 5)
        got = coll.select(lambda c: c.elevation > 5 if c.elevation else False)
        # When elevation==0, the `if` branch evaluates to False, but
        # elevation 0 is not > 5 anyway, so masks should match.
        np.testing.assert_array_equal(got._mask, expected._mask)

    # ------------------------------------------------------------------
    # Validation tests: bad predicates that should raise
    # ------------------------------------------------------------------

    def test_select_scalar_result_raises(self):
        """A predicate that returns a scalar (e.g. .sum()) should raise,
        not silently select one cell or loop forever."""
        coll = RasterCellCollection(self.layer)
        with self.assertRaises(ValueError):
            coll.select(lambda c: c.elevation.sum() > 100)

    def test_select_non_bool_array_raises(self):
        """A predicate returning a numeric (non-bool) array should raise
        during validation, not silently produce wrong results."""
        coll = RasterCellCollection(self.layer)
        with self.assertRaises(ValueError):
            coll.select(lambda c: c.elevation * 2)


class TestRasterCellCollectionData(unittest.TestCase):
    """Tests for get, set, agg, and count methods (Steps 4-5)."""

    def setUp(self) -> None:
        self.model = mesa.Model()
        self.layer = mg.RasterLayer(
            width=3,
            height=4,
            crs="epsg:4326",
            total_bounds=[0, 0, 3, 4],
            model=self.model,
        )
        self.elevation = np.arange(12, dtype=float).reshape(4, 3)
        self.layer.set_band("elevation", self.elevation)

    # ------------------------------------------------------------------
    # get
    # ------------------------------------------------------------------

    def test_get_agreement(self):
        """collection.get(band) matches per-cell cell.<band> reads."""
        # Create a partial mask
        mask = np.zeros((4, 3), dtype=bool)
        mask[1, :] = True
        mask[2, 1] = True
        coll = RasterCellCollection(self.layer, mask)

        arr = coll.get("elevation")
        self.assertIsInstance(arr, np.ndarray)
        self.assertEqual(len(arr), 4)

        cell_vals = [cell.elevation for cell in coll]
        np.testing.assert_array_equal(arr, cell_vals)

    def test_get_multiple_bands(self):
        """get with a list of bands returns a list of arrays."""
        self.layer.set_band("population", self.elevation * 2)
        coll = RasterCellCollection(self.layer)

        res = coll.get(["elevation", "population"])
        self.assertIsInstance(res, list)
        self.assertEqual(len(res), 2)
        np.testing.assert_array_equal(res[0], self.elevation.flatten())
        np.testing.assert_array_equal(res[1], self.elevation.flatten() * 2)

    def test_get_missing_band_error(self):
        """Missing band raises AttributeError by default."""
        coll = RasterCellCollection(self.layer)
        with self.assertRaises(AttributeError):
            coll.get("missing")

    def test_get_missing_band_default(self):
        """Missing band returns default_value array if handle_missing='default'."""
        mask = np.zeros((4, 3), dtype=bool)
        mask[1, :] = True
        coll = RasterCellCollection(self.layer, mask)
        res = coll.get("missing", handle_missing="default", default_value=-99)
        self.assertIsInstance(res, np.ndarray)
        self.assertEqual(len(res), 3)
        np.testing.assert_array_equal(res, [-99, -99, -99])

    def test_get_invalid_handle_missing_raises_immediately(self):
        """Invalid handle_missing raises ValueError even if all bands exist."""
        coll = RasterCellCollection(self.layer)
        with self.assertRaises(ValueError):
            coll.get("elevation", handle_missing="ignore")

    # ------------------------------------------------------------------
    # set
    # ------------------------------------------------------------------

    def test_set_round_trip(self):
        """set() is visible through Cell reads, and Cell write through get()."""
        mask = np.zeros((4, 3), dtype=bool)
        mask[0, 0] = True
        mask[2, 2] = True
        coll = RasterCellCollection(self.layer, mask)

        # 1. set() -> Cell read
        coll.set("elevation", 999.0)
        cells = list(coll)
        self.assertEqual(cells[0].elevation, 999.0)
        self.assertEqual(cells[1].elevation, 999.0)

        # 2. Cell write -> get()
        cells[0].elevation = 888.0
        cells[1].elevation = 777.0
        arr = coll.get("elevation")
        np.testing.assert_array_equal(arr, [888.0, 777.0])

    def test_set_array(self):
        """set() accepts an array of the same length."""
        mask = np.zeros((4, 3), dtype=bool)
        mask[1, :] = True
        coll = RasterCellCollection(self.layer, mask)

        coll.set("elevation", np.array([10.0, 20.0, 30.0]))
        np.testing.assert_array_equal(coll.get("elevation"), [10.0, 20.0, 30.0])

    def test_set_callable(self):
        """set() accepts a callable applied to current values."""
        coll = RasterCellCollection(self.layer)
        coll.set("elevation", lambda x: x + 100.0)
        np.testing.assert_array_equal(
            coll.get("elevation"), self.elevation.flatten() + 100.0
        )

    def test_set_missing_band_raises(self):
        """Setting a non-existent band raises AttributeError."""
        coll = RasterCellCollection(self.layer)
        with self.assertRaises(AttributeError):
            coll.set("missing", 5)

    def test_set_returns_self(self):
        """set() returns the collection for chaining."""
        coll = RasterCellCollection(self.layer)
        res = coll.set("elevation", 5)
        self.assertIs(res, coll)

    # ------------------------------------------------------------------
    # agg and count
    # ------------------------------------------------------------------

    def test_agg_single_callable(self):
        coll = RasterCellCollection(self.layer)
        res = coll.agg("elevation", np.sum)
        self.assertEqual(res, self.elevation.sum())

    def test_agg_iterable_callables(self):
        coll = RasterCellCollection(self.layer)
        res = coll.agg("elevation", [np.min, np.max])
        self.assertIsInstance(res, tuple)
        self.assertEqual(res, (0.0, 11.0))

    def test_agg_empty_selection_raises(self):
        """agg on an empty selection raises ValueError."""
        mask = np.zeros((4, 3), dtype=bool)
        coll = RasterCellCollection(self.layer, mask)
        with self.assertRaises(ValueError):
            coll.agg("elevation", np.mean)

    def test_count(self):
        mask = np.zeros((4, 3), dtype=bool)
        mask[0, 0] = True
        mask[1, 1] = True
        coll = RasterCellCollection(self.layer, mask)
        self.assertEqual(coll.count(), 2)
        self.assertEqual(coll.count(), len(coll))


if __name__ == "__main__":
    unittest.main()
