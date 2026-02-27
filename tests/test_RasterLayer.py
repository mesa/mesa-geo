import os
import tempfile
import unittest
import warnings

import mesa
import numpy as np
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

    def test_get_random_xy_selector_equivalence(self):
        cell = self.raster_layer.cells[1][2]

        self.model.random.seed(19)
        xy_from_cell = self.raster_layer.get_random_xy(cell=cell)
        self.model.random.seed(19)
        xy_from_pos = self.raster_layer.get_random_xy(pos=(1, 2))
        self.model.random.seed(19)
        xy_from_rowcol = self.raster_layer.get_random_xy(rowcol=(0, 1))

        self.assertEqual(xy_from_cell, xy_from_pos)
        self.assertEqual(xy_from_pos, xy_from_rowcol)

    def test_get_random_xy_raises_for_out_of_bounds_pos(self):
        with self.assertRaises(ValueError):
            self.raster_layer.get_random_xy(pos=(self.raster_layer.width, 0))

    def test_get_random_xy_raises_for_out_of_bounds_rowcol(self):
        with self.assertRaises(ValueError):
            self.raster_layer.get_random_xy(rowcol=(self.raster_layer.height, 0))

    def test_get_random_xy_raises_for_cell_without_rowcol(self):
        detached_cell = mg.Cell(self.model, pos=(0, 0), rowcol=None)
        with self.assertRaises(ValueError):
            self.raster_layer.get_random_xy(cell=detached_cell)

    def test_get_random_xy_raises_for_cell_with_out_of_bounds_rowcol(self):
        detached_cell = mg.Cell(
            self.model, pos=(0, 0), rowcol=(self.raster_layer.height, 0)
        )
        with self.assertRaises(ValueError):
            self.raster_layer.get_random_xy(cell=detached_cell)

    def test_out_of_bounds_accepts_pos_rowcol_and_xy(self):
        self.assertFalse(self.raster_layer.out_of_bounds((0, 0)))
        self.assertTrue(self.raster_layer.out_of_bounds((self.raster_layer.width, 0)))

        self.assertFalse(self.raster_layer.out_of_bounds(rowcol=(0, 0)))
        self.assertTrue(
            self.raster_layer.out_of_bounds(rowcol=(self.raster_layer.height, 0))
        )

        min_x, min_y, max_x, max_y = self.raster_layer.total_bounds
        self.assertFalse(self.raster_layer.out_of_bounds(xy=(min_x, min_y)))
        self.assertFalse(self.raster_layer.out_of_bounds(xy=(min_x, max_y)))
        self.assertFalse(self.raster_layer.out_of_bounds(xy=(max_x, min_y)))
        self.assertFalse(self.raster_layer.out_of_bounds(xy=(max_x, max_y)))

        delta_x, delta_y = self.raster_layer.resolution
        self.assertTrue(self.raster_layer.out_of_bounds(xy=(max_x + delta_x, min_y)))
        self.assertTrue(self.raster_layer.out_of_bounds(xy=(min_x, min_y - delta_y)))

    def test_out_of_bounds_raises_for_invalid_selector_combinations(self):
        with self.assertRaises(ValueError):
            self.raster_layer.out_of_bounds()

        with self.assertRaises(ValueError):
            self.raster_layer.out_of_bounds((0, 0), rowcol=(0, 0))

        with self.assertRaises(ValueError):
            self.raster_layer.out_of_bounds((0, 0), xy=self.raster_layer.cells[0][0].xy)

    def test_out_of_bounds_xy_respects_sheared_transform_coverage(self):
        sheared_layer = mg.RasterLayer(
            width=2,
            height=2,
            crs="epsg:4326",
            total_bounds=[0.0, 0.0, 4.0, 2.0],
            model=self.model,
        )
        sheared_layer._transform = rio.transform.Affine(1.0, 1.0, 0.0, 0.0, 1.0, 0.0)

        # Inside bbox but outside the sheared raster footprint.
        self.assertTrue(sheared_layer.out_of_bounds(xy=(0.5, 1.5)))
        # Inside the raster footprint.
        self.assertFalse(sheared_layer.out_of_bounds(xy=(1.5, 0.5)))

    def test_out_of_bounds_xy_rejects_non_finite_values(self):
        self.assertTrue(self.raster_layer.out_of_bounds(xy=(np.nan, 0.0)))
        self.assertTrue(self.raster_layer.out_of_bounds(xy=(0.0, np.nan)))
        self.assertTrue(self.raster_layer.out_of_bounds(xy=(np.inf, 0.0)))
        self.assertTrue(self.raster_layer.out_of_bounds(xy=(0.0, -np.inf)))

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
