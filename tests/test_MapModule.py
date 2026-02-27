import base64
import unittest
from io import BytesIO
from unittest.mock import patch

import mesa
import numpy as np
import xyzservices.providers as xyz
from ipyleaflet import Circle, CircleMarker, Marker
from PIL import Image
from shapely.geometry import LineString, Point, Polygon

import mesa_geo as mg
import mesa_geo.visualization as mgv


class TestMapModule(unittest.TestCase):
    def setUp(self) -> None:
        self.model = mesa.Model()
        self.model.space = mg.GeoSpace(crs="epsg:4326")
        self.agent_creator = mg.AgentCreator(
            agent_class=mg.GeoAgent, model=self.model, crs="epsg:4326"
        )
        self.points = [Point(1, 1)] * 7
        self.point_agents = [
            self.agent_creator.create_agent(point) for point in self.points
        ]
        self.lines = [LineString([(1, 1), (2, 2)])] * 9
        self.line_agents = [
            self.agent_creator.create_agent(line) for line in self.lines
        ]
        self.polygons = [Polygon([(1, 1), (2, 2), (4, 4)])] * 3
        self.polygon_agents = [
            self.agent_creator.create_agent(polygon) for polygon in self.polygons
        ]
        self.raster_layer = mg.RasterLayer(
            1, 1, crs="epsg:4326", total_bounds=[0, 0, 1, 1], model=self.model
        )
        self.raster_layer.apply_raster(np.array([[[0]]]))

    def tearDown(self) -> None:
        pass

    def test_render_point_agents(self):
        # test length point agents and Circle marker as default
        map_module = mgv.MapModule(
            portrayal_method=lambda x: {"color": "Green"},
            tiles=xyz.OpenStreetMap.Mapnik,
        )
        self.model.space.add_agents(self.point_agents)
        self.assertEqual(len(map_module.render(self.model).get("agents")[1]), 7)
        self.assertIsInstance(map_module.render(self.model).get("agents")[1][3], Circle)
        # test CircleMarker option
        map_module = mgv.MapModule(
            portrayal_method=lambda x: {
                "marker_type": "CircleMarker",
                "color": "Green",
            },
            tiles=xyz.OpenStreetMap.Mapnik,
        )
        self.model.space.add_agents(self.point_agents)
        self.assertIsInstance(
            map_module.render(self.model).get("agents")[1][3], CircleMarker
        )

        # test Marker option
        map_module = mgv.MapModule(
            portrayal_method=lambda x: {
                "marker_type": "AwesomeIcon",
                "name": "bus",
                "color": "Green",
            },
            tiles=xyz.OpenStreetMap.Mapnik,
        )
        self.model.space.add_agents(self.point_agents)
        self.assertEqual(len(map_module.render(self.model).get("agents")[1]), 7)
        self.assertIsInstance(map_module.render(self.model).get("agents")[1][3], Marker)
        # test popupProperties for Point
        map_module = mgv.MapModule(
            portrayal_method=lambda x: {
                "color": "Red",
                "radius": 7,
                "description": "popupMsg",
            },
            tiles=xyz.OpenStreetMap.Mapnik,
        )
        self.model.space.add_agents(self.point_agents)
        print(map_module.render(self.model).get("agents")[0])
        self.assertDictEqual(
            map_module.render(self.model).get("agents")[0],
            {
                "type": "FeatureCollection",
                "features": [] * len(self.point_agents),
            },
        )

        # test ValueError if not known markertype
        map_module = mgv.MapModule(
            portrayal_method=lambda x: {"marker_type": "Hexagon", "color": "Green"},
            tiles=xyz.OpenStreetMap.Mapnik,
        )
        self.model.space.add_agents(self.point_agents)
        with self.assertRaises(ValueError):
            map_module.render(self.model)

    def test_render_line_agents(self):
        map_module = mgv.MapModule(
            portrayal_method=lambda x: {"color": "#3388ff", "weight": 7},
            tiles=xyz.OpenStreetMap.Mapnik,
        )
        self.model.space.add_agents(self.line_agents)
        self.assertDictEqual(
            map_module.render(self.model).get("agents")[0],
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "geometry": {
                            "type": "LineString",
                            "coordinates": ((1.0, 1.0), (2.0, 2.0)),
                        },
                        "properties": {"style": {"color": "#3388ff", "weight": 7}},
                    }
                ]
                * len(self.line_agents),
            },
        )

        map_module = mgv.MapModule(
            portrayal_method=lambda x: {
                "color": "#3388ff",
                "weight": 7,
                "description": "popupMsg",
            },
            tiles=xyz.OpenStreetMap.Mapnik,
        )
        self.model.space.add_agents(self.line_agents)
        self.assertDictEqual(
            map_module.render(self.model).get("agents")[0],
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "geometry": {
                            "type": "LineString",
                            "coordinates": ((1.0, 1.0), (2.0, 2.0)),
                        },
                        "properties": {
                            "style": {"color": "#3388ff", "weight": 7},
                            "popupProperties": "popupMsg",
                        },
                    }
                ]
                * len(self.line_agents),
            },
        )

    def test_render_polygon_agents(self):
        self.maxDiff = None

        map_module = mgv.MapModule(
            portrayal_method=lambda x: {"fillColor": "#3388ff", "fillOpacity": 0.7},
            tiles=xyz.OpenStreetMap.Mapnik,
        )
        self.model.space.add_agents(self.polygon_agents)
        self.assertDictEqual(
            map_module.render(self.model).get("agents")[0],
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": (
                                ((1.0, 1.0), (2.0, 2.0), (4.0, 4.0), (1.0, 1.0)),
                            ),
                        },
                        "properties": {
                            "style": {"fillColor": "#3388ff", "fillOpacity": 0.7}
                        },
                    }
                ]
                * len(self.polygon_agents),
            },
        )

        map_module = mgv.MapModule(
            portrayal_method=lambda x: {
                "fillColor": "#3388ff",
                "fillOpacity": 0.7,
                "description": "popupMsg",
            },
            tiles=xyz.OpenStreetMap.Mapnik,
        )
        self.model.space.add_agents(self.polygon_agents)
        self.assertDictEqual(
            map_module.render(self.model).get("agents")[0],
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": (
                                ((1.0, 1.0), (2.0, 2.0), (4.0, 4.0), (1.0, 1.0)),
                            ),
                        },
                        "properties": {
                            "style": {"fillColor": "#3388ff", "fillOpacity": 0.7},
                            "popupProperties": "popupMsg",
                        },
                    }
                ]
                * len(self.polygon_agents),
            },
        )

    def test_render_raster_layers(self):
        map_module = mgv.MapModule(
            portrayal_method=lambda x: (255, 255, 255, 0.5),
            tiles=xyz.OpenStreetMap.Mapnik,
        )
        self.model.space.add_layer(self.raster_layer)
        self.model.space.add_layer(
            self.raster_layer.to_image(colormap=lambda x: (0, 0, 0, 1))
        )

        # _render_layers should call image_to_url with the correct RGBA arrays
        # and return the expected layer metadata structure.
        # We mock image_to_url so the test does not depend on PNG/zlib byte output.
        captured_arrays = []

        def capture_url(array):
            captured_arrays.append(np.array(array))
            return f"mock://raster/{len(captured_arrays)}"

        with patch(
            "mesa_geo.visualization.components.geospace_component.image_to_url",
            side_effect=capture_url,
        ) as mocked_image_to_url:
            self.assertDictEqual(
                map_module.render(self.model).get("layers"),
                {
                    "rasters": [
                        {
                            "url": "mock://raster/1",
                            "bounds": [[0.0, 0.0], [1.0, 1.0]],
                        },
                        {
                            "url": "mock://raster/2",
                            "bounds": [[0.0, 0.0], [1.0, 1.0]],
                        },
                    ],
                    "total_bounds": [[0.0, 0.0], [1.0, 1.0]],
                    "vectors": [],
                },
            )

        self.assertEqual(mocked_image_to_url.call_count, 2)
        self.assertEqual(captured_arrays[0].shape, (1, 1, 4))
        self.assertEqual(captured_arrays[1].shape, (1, 1, 4))
        np.testing.assert_allclose(
            captured_arrays[0], np.array([[[255.0, 255.0, 255.0, 0.5]]])
        )
        np.testing.assert_allclose(
            captured_arrays[1], np.array([[[0.0, 0.0, 0.0, 1.0]]])
        )

    def test_render_raster_layers_png_data_url_smoke(self):
        map_module = mgv.MapModule(
            portrayal_method=lambda x: (255, 255, 255, 0.5),
            tiles=xyz.OpenStreetMap.Mapnik,
        )
        self.model.space.add_layer(self.raster_layer)
        self.model.space.add_layer(
            self.raster_layer.to_image(colormap=lambda x: (0, 0, 0, 1))
        )

        # with the real encoder, raster URLs should be valid PNG data URLs that
        # decode to expected image semantics.
        rasters = map_module.render(self.model).get("layers")["rasters"]
        # First raster is the RasterLayer rendered via MapModule portrayal_method
        # (255, 255, 255, 0.5), which write_png normalization maps to white + alpha 255.
        # Second raster is the precomputed ImageLayer from to_image(colormap=(0, 0, 0, 1)),
        # which stays black + alpha 255.
        expected_pixels = [(255, 255, 255, 255), (0, 0, 0, 255)]
        for raster, expected_pixel in zip(rasters, expected_pixels, strict=True):
            url = raster["url"]
            self.assertTrue(url.startswith("data:image/png;base64,"))
            payload = url.split(",", maxsplit=1)[1]
            png_bytes = base64.b64decode(payload, validate=True)
            self.assertTrue(png_bytes.startswith(b"\x89PNG\r\n\x1a\n"))
            self.assertIn(b"IHDR", png_bytes)
            self.assertTrue(png_bytes.endswith(b"\x00\x00\x00\x00IEND\xaeB`\x82"))
            with Image.open(BytesIO(png_bytes)) as image:
                self.assertEqual(image.mode, "RGBA")
                self.assertEqual(image.size, (1, 1))
                self.assertEqual(image.getpixel((0, 0)), expected_pixel)
