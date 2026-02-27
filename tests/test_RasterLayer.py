import os
import tempfile
import unittest
import warnings

import mesa
import numpy as np
import pytest
import rasterio as rio

import mesa_geo as mg


class TestRasterLayer(unittest.TestCase):
    def setUp(self) -> None:
        self.model = mesa.Model()
        self.raster_layer = mg.RasterLayer(
            width=2,
            height=3,
            crs="epsg:4326",
            total_bounds=[
                -122.26638888878,
                42.855833333,
                -121.94972222209202,
                43.01472222189958,
            ],
            model=self.model,
        )
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmpdir = self._tmpdir.name
        self._setup_raster_files()

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def _setup_raster_files(self) -> None:
        self.multi_band_values = np.array(
            [
                [[1, 2], [3, 4]],
                [[10, 20], [30, 40]],
            ]
        )
        self.single_band_values = np.array([[[9, 8], [7, 6]]])
        transform = rio.transform.from_bounds(-1, -1, 1, 1, 2, 2)

        self.multi_band_path = os.path.join(self.tmpdir, "multi_band.tif")
        with rio.open(
            self.multi_band_path,
            "w",
            driver="GTiff",
            width=2,
            height=2,
            count=2,
            dtype=self.multi_band_values.dtype,
            crs="epsg:4326",
            transform=transform,
        ) as dataset:
            dataset.write(self.multi_band_values)

        self.single_band_path = os.path.join(self.tmpdir, "single_band.tif")
        with rio.open(
            self.single_band_path,
            "w",
            driver="GTiff",
            width=2,
            height=2,
            count=1,
            dtype=self.single_band_values.dtype,
            crs="epsg:4326",
            transform=transform,
        ) as dataset:
            dataset.write(self.single_band_values)

    def test_apply_raster(self):
        raster_data = np.array([[[1, 2], [3, 4], [5, 6]]])
        self.raster_layer.apply_raster(raster_data, attr_name="val")
        """
        (x, y) coordinates:
        (0, 2), (1, 2)
        (0, 1), (1, 1)
        (0, 0), (1, 0)

        values:
        [[[1, 2],
          [3, 4],
          [5, 6]]]
        """
        self.assertEqual(self.raster_layer.cells[0][1].val, 3)
        self.assertEqual(self.raster_layer.attributes, {"val"})

        self.raster_layer.apply_raster(raster_data, attr_name="elevation")
        self.assertEqual(self.raster_layer.cells[0][1].elevation, 3)
        self.assertEqual(self.raster_layer.attributes, {"val", "elevation"})

        with self.assertRaises(ValueError):
            self.raster_layer.apply_raster(np.empty((1, 100, 100)))

    def test_apply_raster_single_band_attr_name_none(self):
        raster_data = np.array([[[7, 8], [9, 10], [11, 12]]])
        self.raster_layer.apply_raster(raster_data)

        self.assertEqual(len(self.raster_layer.attributes), 1)
        np.testing.assert_array_equal(self.raster_layer.get_raster(), raster_data)

    def test_apply_raster_single_band_attr_name_list(self):
        raster_data = np.array([[[7, 8], [9, 10], [11, 12]]])
        self.raster_layer.apply_raster(raster_data, attr_name=["elevation"])

        self.assertEqual(self.raster_layer.attributes, {"elevation"})
        np.testing.assert_array_equal(
            self.raster_layer.get_raster(attr_name="elevation"), raster_data
        )

    def test_apply_raster_single_band_attr_name_list_mismatch(self):
        raster_data = np.array([[[7, 8], [9, 10], [11, 12]]])
        with self.assertRaises(ValueError):
            self.raster_layer.apply_raster(
                raster_data, attr_name=["elevation", "water_level"]
            )

    def test_apply_raster_multiband_attr_name_list(self):
        raster_data = np.array(
            [
                [[1, 2], [3, 4], [5, 6]],
                [[10, 20], [30, 40], [50, 60]],
            ]
        )
        self.raster_layer.apply_raster(
            raster_data, attr_name=["elevation", "water_level"]
        )

        np.testing.assert_array_equal(
            self.raster_layer.get_raster(attr_name="elevation"), raster_data[0:1]
        )
        np.testing.assert_array_equal(
            self.raster_layer.get_raster(attr_name="water_level"), raster_data[1:2]
        )

    def test_apply_raster_multiband_attr_name_none(self):
        raster_data = np.array(
            [
                [[1, 2], [3, 4], [5, 6]],
                [[10, 20], [30, 40], [50, 60]],
            ]
        )
        self.raster_layer.apply_raster(raster_data)

        data = self.raster_layer.get_raster()
        self.assertEqual(data.shape, raster_data.shape)
        self.assertTrue(
            any(
                np.array_equal(data[idx], raster_data[0])
                for idx in range(data.shape[0])
            )
        )
        self.assertTrue(
            any(
                np.array_equal(data[idx], raster_data[1])
                for idx in range(data.shape[0])
            )
        )

    def test_apply_raster_multiband_attr_name_string(self):
        raster_data = np.array(
            [
                [[1, 2], [3, 4], [5, 6]],
                [[10, 20], [30, 40], [50, 60]],
            ]
        )
        self.raster_layer.apply_raster(raster_data, attr_name="band")

        self.assertEqual(self.raster_layer.attributes, {"band_1", "band_2"})
        np.testing.assert_array_equal(
            self.raster_layer.get_raster(attr_name="band_1"), raster_data[0:1]
        )
        np.testing.assert_array_equal(
            self.raster_layer.get_raster(attr_name="band_2"), raster_data[1:2]
        )

    def test_apply_raster_multiband_attr_name_list_mismatch(self):
        raster_data = np.array(
            [
                [[1, 2], [3, 4], [5, 6]],
                [[10, 20], [30, 40], [50, 60]],
            ]
        )
        with self.assertRaises(ValueError):
            self.raster_layer.apply_raster(raster_data, attr_name=["only_one"])

    def test_get_raster(self):
        raster_data = np.array([[[1, 2], [3, 4], [5, 6]]])
        self.raster_layer.apply_raster(raster_data, attr_name="val")
        """
        (x, y) coordinates:
        (0, 2), (1, 2)
        (0, 1), (1, 1)
        (0, 0), (1, 0)

        values:
        [[[1, 2],
          [3, 4],
          [5, 6]]]
        """
        self.raster_layer.apply_raster(raster_data, attr_name="elevation")
        np.testing.assert_array_equal(
            self.raster_layer.get_raster(attr_name="elevation"), raster_data
        )

        self.raster_layer.apply_raster(raster_data)
        data = self.raster_layer.get_raster()
        self.assertEqual(data.shape, (3, 3, 2))
        for band in data:
            np.testing.assert_array_equal(band, raster_data[0])
        with self.assertRaises(ValueError):
            self.raster_layer.get_raster("not_existing_attr")

    def test_get_raster_attr_name_list(self):
        raster_data = np.array(
            [
                [[1, 2], [3, 4], [5, 6]],
                [[10, 20], [30, 40], [50, 60]],
            ]
        )
        self.raster_layer.apply_raster(
            raster_data, attr_name=["elevation", "water_level"]
        )
        np.testing.assert_array_equal(
            self.raster_layer.get_raster(attr_name=["water_level", "elevation"]),
            np.array([raster_data[1], raster_data[0]]),
        )

    def test_get_raster_attr_name_list_missing(self):
        raster_data = np.array(
            [
                [[1, 2], [3, 4], [5, 6]],
                [[10, 20], [30, 40], [50, 60]],
            ]
        )
        self.raster_layer.apply_raster(
            raster_data, attr_name=["elevation", "water_level"]
        )
        with self.assertRaises(ValueError):
            self.raster_layer.get_raster(attr_name=["elevation", "missing"])

    def test_to_file_attr_name_list(self):
        raster_data = np.array(
            [
                [[1, 2], [3, 4], [5, 6]],
                [[10, 20], [30, 40], [50, 60]],
            ]
        )
        self.raster_layer.apply_raster(
            raster_data, attr_name=["elevation", "water_level"]
        )

        path = os.path.join(self.tmpdir, "selected_bands.tif")
        self.raster_layer.to_file(path, attr_name=["water_level", "elevation"])

        with rio.open(path, "r") as dataset:
            values = dataset.read()

        np.testing.assert_array_equal(values[0], raster_data[1])
        np.testing.assert_array_equal(values[1], raster_data[0])

    def test_get_min_cell(self):
        self.raster_layer.apply_raster(
            np.array([[[1, 2], [3, 4], [5, 6]]]), attr_name="elevation"
        )

        min_cell = min(
            self.raster_layer.get_neighboring_cells(pos=(0, 2), moore=True),
            key=lambda cell: cell.elevation,
        )
        self.assertEqual(min_cell.pos, (1, 2))
        self.assertEqual(min_cell.elevation, 2)

        min_cell = min(
            self.raster_layer.get_neighboring_cells(
                pos=(0, 2), moore=True, include_center=True
            ),
            key=lambda cell: cell.elevation,
        )
        self.assertEqual(min_cell.pos, (0, 2))
        self.assertEqual(min_cell.elevation, 1)

        self.raster_layer.apply_raster(
            np.array([[[1, 2], [3, 4], [5, 6]]]), attr_name="water_level"
        )
        min_cell = min(
            self.raster_layer.get_neighboring_cells(
                pos=(0, 2), moore=True, include_center=True
            ),
            key=lambda cell: cell.elevation + cell.water_level,
        )
        self.assertEqual(min_cell.pos, (0, 2))
        self.assertEqual(min_cell.elevation, 1)
        self.assertEqual(min_cell.water_level, 1)

    def test_get_max_cell(self):
        self.raster_layer.apply_raster(
            np.array([[[1, 2], [3, 4], [5, 6]]]), attr_name="elevation"
        )

        max_cell = max(
            self.raster_layer.get_neighboring_cells(pos=(0, 2), moore=True),
            key=lambda cell: cell.elevation,
        )
        self.assertEqual(max_cell.pos, (1, 1))
        self.assertEqual(max_cell.elevation, 4)

    def test_cell_incompatible_assignment(self):
        self.raster_layer.apply_raster(
            np.array([[[1, 2], [3, 4], [5, 6]]]), attr_name="elevation"
        )
        cell = self.raster_layer.cells[0][0]
        # Valid assignment updates backend
        cell.elevation = 99
        self.assertEqual(cell.elevation, 99)
        # Incompatible assignment skips backend but doesn't raise
        cell.elevation = None
        self.assertIsNone(cell.__dict__.get("elevation"))

    def test_deprecated_pos_indices_accessors(self):
        cell = self.raster_layer.cells[0][0]
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            self.assertEqual(cell.indices, (2, 0))
        self.assertEqual(len(captured), 1)
        self.assertTrue(
            all(issubclass(item.category, DeprecationWarning) for item in captured)
        )
        self.assertIn("Cell.indices is deprecated", str(captured[0].message))

    def test_transform_accuracy(self):
        """
        Verify that cell.xy and cell.rowcol are calculated correctly.
        """
        # Bottom-Left (grid=0,0) -> Array Row=2, Col=0
        bl_cell = self.raster_layer.cells[0][0]
        self.assertEqual(bl_cell.pos, (0, 0))
        self.assertEqual(bl_cell.rowcol, (2, 0))

        # Transform logic: x_coord, y_coord = transform * (col + 0.5, row + 0.5)
        expected_x, expected_y = self.raster_layer.transform * (0.5, 2.5)
        self.assertAlmostEqual(bl_cell.xy[0], expected_x)
        self.assertAlmostEqual(bl_cell.xy[1], expected_y)

        # Top-Right (grid=1,2) -> Array Row=0, Col=1
        tr_cell = self.raster_layer.cells[1][2]
        self.assertEqual(tr_cell.pos, (1, 2))
        self.assertEqual(tr_cell.rowcol, (0, 1))

        expected_xy = rio.transform.xy(
            self.raster_layer.transform, 0, 1, offset="center"
        )
        self.assertEqual(tr_cell.xy, expected_xy)

    def test_cell_xy_updates_after_to_crs(self):
        original_xy = self.raster_layer.cells[0][0].xy
        transformed_layer = self.raster_layer.to_crs("epsg:3857")
        transformed_cell = transformed_layer.cells[0][0]
        expected_xy = rio.transform.xy(
            transformed_layer.transform, *transformed_cell.rowcol, offset="center"
        )
        self.assertEqual(transformed_cell.xy, expected_xy)
        self.assertEqual(self.raster_layer.cells[0][0].xy, original_xy)
        self.assertNotEqual(transformed_cell.xy, original_xy)

    def test_from_file_multiband_attr_name_list(self):
        layer = mg.RasterLayer.from_file(
            self.multi_band_path, self.model, attr_name=["band_1", "band_2"]
        )

        self.assertEqual(layer.attributes, {"band_1", "band_2"})
        np.testing.assert_array_equal(
            layer.get_raster(attr_name="band_1"), self.multi_band_values[0:1]
        )
        np.testing.assert_array_equal(
            layer.get_raster(attr_name="band_2"), self.multi_band_values[1:2]
        )

    def test_from_file_multiband_attr_name_base(self):
        layer = mg.RasterLayer.from_file(
            self.multi_band_path, self.model, attr_name="band"
        )

        self.assertEqual(layer.attributes, {"band_1", "band_2"})
        np.testing.assert_array_equal(
            layer.get_raster(attr_name="band_1"), self.multi_band_values[0:1]
        )
        np.testing.assert_array_equal(
            layer.get_raster(attr_name="band_2"), self.multi_band_values[1:2]
        )

    def test_from_file_multiband_attr_name_length_mismatch(self):
        with self.assertRaises(ValueError):
            mg.RasterLayer.from_file(
                self.multi_band_path, self.model, attr_name=["only_one"]
            )

    def test_from_file_single_band_attr_name_list(self):
        layer = mg.RasterLayer.from_file(
            self.single_band_path, self.model, attr_name=["elevation"]
        )

        self.assertEqual(layer.attributes, {"elevation"})
        np.testing.assert_array_equal(
            layer.get_raster(attr_name="elevation"), self.single_band_values
        )

    def test_from_file_single_band_attr_name_string(self):
        layer = mg.RasterLayer.from_file(
            self.single_band_path, self.model, attr_name="elevation"
        )

        self.assertEqual(layer.attributes, {"elevation"})
        np.testing.assert_array_equal(
            layer.get_raster(attr_name="elevation"), self.single_band_values
        )

    def test_from_file_single_band_attr_name_none(self):
        layer = mg.RasterLayer.from_file(self.single_band_path, self.model)

        self.assertEqual(len(layer.attributes), 1)
        np.testing.assert_array_equal(layer.get_raster(), self.single_band_values)

    def test_from_file_single_band_attr_name_length_mismatch(self):
        with self.assertRaises(ValueError):
            mg.RasterLayer.from_file(
                self.single_band_path, self.model, attr_name=["a", "b"]
            )

    def test_from_file_multiband_attr_name_none(self):
        layer = mg.RasterLayer.from_file(self.multi_band_path, self.model)

        data = layer.get_raster()
        self.assertEqual(data.shape, self.multi_band_values.shape)
        self.assertTrue(
            any(
                np.array_equal(data[idx], self.multi_band_values[0])
                for idx in range(data.shape[0])
            )
        )
        self.assertTrue(
            any(
                np.array_equal(data[idx], self.multi_band_values[1])
                for idx in range(data.shape[0])
            )
        )


def test_cell_missing_raster_layer_kwarg():
    class OldSchoolCell(mg.Cell):
        def __init__(self, model, pos=None, indices=None):
            super().__init__(model, pos, indices)

    rl = mg.RasterLayer(
        10,
        10,
        "epsg:4326",
        total_bounds=[0, 0, 10, 10],
        model=mesa.Model(),
        cell_cls=OldSchoolCell,
    )
    cell = rl.cells[0][0]
    assert isinstance(cell, OldSchoolCell)
    assert cell.raster_layer is rl


def test_cell_wrapper_dunder_methods():
    rl = mg.RasterLayer(
        10, 10, "epsg:4326", total_bounds=[0, 0, 10, 10], model=mesa.Model()
    )

    # Test __iter__ of _CellWrapper
    cols = list(rl.cells)
    assert len(cols) == 10

    # Test __iter__ and __len__ of _CellColumn
    col = rl.cells[0]
    cells = list(col)
    assert len(cells) == 10
    assert len(col) == 10

    # Test exceptions
    with pytest.raises(TypeError):
        _ = rl.cells["invalid"]

    with pytest.raises(TypeError):
        _ = rl.cells[0]["invalid"]


def test_cell_coverage():
    rl = mg.RasterLayer(
        10, 10, "epsg:4326", total_bounds=[0, 0, 10, 10], model=mesa.Model()
    )
    rl.apply_raster(np.ones((1, 10, 10), dtype=np.int32), attr_name="test_attr")

    wrapper = rl.cells
    cell1 = wrapper[0][0]
    cell2 = wrapper[0][0]  # Hit cache
    assert cell1 is cell2

    with pytest.raises(AttributeError):
        _ = cell1.non_existent_attr

    # Hit TypeError fallback in Cell.__setattr__ by assigning None to numpy numeric matrix
    cell1.test_attr = None
    assert cell1.__dict__["test_attr"] is None

    # Hit the pos is None branch
    cell1._pos = None
    cell1.test_attr = 5
    assert cell1.__dict__["test_attr"] == 5

    cell1.step()

    with pytest.warns(DeprecationWarning):
        rl._sync_cell_xy()


def test_cell_extra_coverage():
    # Hit missing random assignment coverage
    cell = mg.Cell(model=mesa.Model(), pos=(5, 5))
    assert hasattr(cell, "random")

    # Test xy property returns None when raster_layer or rowcol is None
    assert cell.xy is None


def test_raster_layer_getitem():
    rl = mg.RasterLayer(
        10, 10, "epsg:4326", total_bounds=[0, 0, 10, 10], model=mesa.Model()
    )

    # 1. Test int indexing: rl[x] -> returns list of cells
    col = rl[3]
    assert len(col) == 10
    assert col[5].pos == (3, 5)

    # 2. Test sequence of coordinates: rl[[(0,0), (1,1)]] -> returns list of cells
    cells = rl[[(0, 0), (1, 1)]]
    assert len(cells) == 2
    assert cells[0].pos == (0, 0)
    assert cells[1].pos == (1, 1)

    # 3. Test grid indexing: rl[x, y] -> returns single cell
    cell = rl[3, 5]
    assert cell.pos == (3, 5)

    # 4. Test slice indexing: rl[3:5, 4:6]
    # Note from code: rl[slice, slice] loops over the slices
    cells_slice = rl[3:5, 4:6]
    # Expect generator/list of cells
    cells_list = list(cells_slice)
    assert len(cells_list) > 0
