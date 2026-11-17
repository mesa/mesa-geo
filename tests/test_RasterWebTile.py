import unittest

import xyzservices.providers as xyz

import mesa_geo as mg


class TestRasterWebTile(unittest.TestCase):
    def test_from_xyzservices(self):
        provider = xyz.OpenStreetMap.Mapnik
        map_tile = mg.RasterWebTile.from_xyzservices(provider)

        self.assertEqual(map_tile.url, provider.url)
        self.assertEqual(map_tile.kind, "raster_web_tile")
        self.assertEqual(map_tile.options["name"], provider.name)
        self.assertEqual(map_tile.options["max_zoom"], provider.max_zoom)
        self.assertEqual(map_tile.options["attribution"], provider.html_attribution)
        self.assertNotIn("url", map_tile.options)
        self.assertNotIn("html_attribution", map_tile.options)
        self.assertEqual(
            map_tile.to_dict(),
            {
                "url": "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
                "options": {
                    "max_zoom": 19,
                    "attribution": '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
                    "name": "OpenStreetMap.Mapnik",
                },
                "kind": "raster_web_tile",
            },
        )

    def test_to_dict(self):
        map_tile = mg.RasterWebTile(
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
            options={
                "attribution": "Map data © <a href='https://openstreetmap.org'>OpenStreetMap</a> contributors"
            },
        )
        self.assertEqual(
            map_tile.to_dict(),
            {
                "kind": "raster_web_tile",
                "url": "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
                "options": {
                    "attribution": "Map data © <a href='https://openstreetmap.org'>OpenStreetMap</a> contributors",
                },
            },
        )

        map_tile = mg.RasterWebTile(
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        )
        self.assertEqual(
            map_tile.to_dict(),
            {
                "kind": "raster_web_tile",
                "url": "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
                "options": None,
            },
        )
