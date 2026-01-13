import unittest

import mesa
import numpy as np

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

    def tearDown(self) -> None:
        pass

    def test_apple_raster(self):
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
        # We expect 3 layers: val, elevation, and the new unnamed one.
        # Since they are all identical raster_data, the order doesn't matter for equality check.
        np.testing.assert_array_equal(
            self.raster_layer.get_raster(),
            np.concatenate((raster_data, raster_data, raster_data)),
        )
        with self.assertRaises(ValueError):
            self.raster_layer.get_raster("not_existing_attr")

    def test_get_min_cell(self):
        self.raster_layer.apply_raster(
            np.array([[[1, 2], [3, 4], [5, 6]]]), attr_name="elevation"
        )

        min_cell = min(
            self.raster_layer.get_neighboring_cells(pos=(0, 2), moore=True),
            key=lambda cell: cell.elevation,
        )
        self.assertEqual(min_cell.grid_pos, (1, 2))
        self.assertEqual(min_cell.elevation, 2)

        min_cell = min(
            self.raster_layer.get_neighboring_cells(
                pos=(0, 2), moore=True, include_center=True
            ),
            key=lambda cell: cell.elevation,
        )
        self.assertEqual(min_cell.grid_pos, (0, 2))
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
        self.assertEqual(min_cell.grid_pos, (0, 2))
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
        self.assertEqual(max_cell.grid_pos, (1, 1))
        self.assertEqual(max_cell.elevation, 4)

    def test_cell_coordinates_and_deprecation(self):
        """
        Verify that:
        1. Accessing cell.pos and cell.indices raises FutureWarning.
        2. cell.pos returns grid_pos and cell.indices returns rowcol.
        """
        cell = self.raster_layer.cells[0][0]

        # Test deprecated 'pos'
        with self.assertWarns(FutureWarning):
            pos = cell.pos
        self.assertEqual(pos, cell.grid_pos)

        # Test deprecated 'indices'
        with self.assertWarns(FutureWarning):
            indices = cell.indices
        self.assertEqual(indices, cell.rowcol)

        # Test setter warning
        with self.assertWarns(FutureWarning):
            cell.pos = (10, 10)
        self.assertEqual(cell.grid_pos, (10, 10))

        with self.assertWarns(FutureWarning):
            cell.indices = (5, 5)
        self.assertEqual(cell.rowcol, (5, 5))

    def test_transform_accuracy(self):
        """
        Verify that cell.xy and cell.rowcol are calculated correctly.
        """
        # Bottom-Left (grid=0,0) -> Array Row=2, Col=0
        bl_cell = self.raster_layer.cells[0][0]
        self.assertEqual(bl_cell.grid_pos, (0, 0))
        self.assertEqual(bl_cell.rowcol, (2, 0))

        # Transform logic: x_coord, y_coord = transform * (col + 0.5, row + 0.5)
        expected_x, expected_y = self.raster_layer.transform * (0.5, 2.5)
        self.assertAlmostEqual(bl_cell.xy[0], expected_x)
        self.assertAlmostEqual(bl_cell.xy[1], expected_y)

        # Top-Right (grid=1,2) -> Array Row=0, Col=1
        tr_cell = self.raster_layer.cells[1][2]
        self.assertEqual(tr_cell.grid_pos, (1, 2))
        self.assertEqual(tr_cell.rowcol, (0, 1))

        expected_x, expected_y = self.raster_layer.transform * (1.5, 0.5)
        self.assertAlmostEqual(tr_cell.xy[0], expected_x)
        self.assertAlmostEqual(tr_cell.xy[1], expected_y)
