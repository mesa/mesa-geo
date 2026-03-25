import numpy as np
from mesa import Model

from mesa_geo.raster_layers import RasterLayer


class DummyModel(Model):
    pass


def test_property_layer_basic():

    model = DummyModel()

    raster = RasterLayer(
        width=5,
        height=5,
        crs="EPSG:4326",
        total_bounds=[0, 0, 5, 5],
        model=model,
    )

    data = np.ones((5, 5))

    raster.apply_raster(data, "temperature")

    # Fixed assertion based on PropertyLayer naming convention
    assert "temperature_0" in raster.property_layer.attributes


def test_property_layer_multiband():

    model = DummyModel()

    raster = RasterLayer(
        width=5,
        height=5,
        crs="EPSG:4326",
        total_bounds=[0, 0, 5, 5],
        model=model,
    )

    data = np.ones((2, 5, 5))

    raster.apply_raster(data, "band")

    assert len(raster.property_layer.attributes) == 2


def test_cell_attribute():

    model = DummyModel()

    raster = RasterLayer(
        width=5,
        height=5,
        crs="EPSG:4326",
        total_bounds=[0, 0, 5, 5],
        model=model,
    )

    data = np.ones((5, 5))

    raster.apply_raster(data, "temperature")

    cell = raster.cells[0][0]

    assert hasattr(cell, "temperature_0")


def test_get_raster():

    model = DummyModel()

    raster = RasterLayer(
        width=5,
        height=5,
        crs="EPSG:4326",
        total_bounds=[0, 0, 5, 5],
        model=model,
    )

    data = np.ones((5, 5))

    raster.apply_raster(data, "temperature")

    result = raster.get_raster("temperature_0")

    assert result is not None
