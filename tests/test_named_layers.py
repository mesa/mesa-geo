import geopandas as gpd
import mesa
import numpy as np
import pytest
import rasterio as rio
from shapely.geometry import Point

from mesa_geo import Cell, GeoSpace
from mesa_geo.raster_layers import ImageLayer, RasterLayer


def test_add_get_roundtrip():
    space = GeoSpace()
    layer1 = ImageLayer(
        values=np.zeros((1, 10, 10)), crs="epsg:4326", total_bounds=[0, 0, 10, 10]
    )
    space.add_layer(layer1, name="layer1")
    assert space.get_layer("layer1") is layer1


def test_unknown_name():
    space = GeoSpace()
    layer1 = ImageLayer(
        values=np.zeros((1, 10, 10)), crs="epsg:4326", total_bounds=[0, 0, 10, 10]
    )
    space.add_layer(layer1, name="layer1")
    with pytest.raises(
        KeyError, match=r"No layer named 'layer2'. Available names: \['layer1'\]"
    ):
        space.get_layer("layer2")


def test_duplicate_name():
    space = GeoSpace()
    layer1 = ImageLayer(
        values=np.zeros((1, 10, 10)), crs="epsg:4326", total_bounds=[0, 0, 10, 10]
    )
    layer2 = ImageLayer(
        values=np.zeros((1, 10, 10)), crs="epsg:4326", total_bounds=[0, 0, 10, 10]
    )
    space.add_layer(layer1, name="layer1")
    with pytest.raises(
        ValueError, match="A layer named 'layer1' is already registered"
    ):
        space.add_layer(layer2, name="layer1")
    # Validation must not mutate layer2's CRS prior to rejection
    assert layer2.crs.to_string() == "EPSG:4326"


def test_layers_list_semantics():
    space = GeoSpace()
    layer1 = ImageLayer(
        values=np.zeros((1, 10, 10)), crs="epsg:4326", total_bounds=[0, 0, 10, 10]
    )
    layer2 = ImageLayer(
        values=np.zeros((1, 10, 10)), crs="epsg:4326", total_bounds=[0, 0, 10, 10]
    )
    space.add_layer(layer1, name="first")
    space.add_layer(layer2)
    assert len(space.layers) == 2
    assert space.layers[0] is layer1
    assert space.layers[1] is layer2
    assert list(space.layers) == [layer1, layer2]


def test_add_layer_without_name():
    space = GeoSpace()
    layer1 = ImageLayer(
        values=np.zeros((1, 10, 10)), crs="epsg:4326", total_bounds=[0, 0, 10, 10]
    )
    space.add_layer(layer1)
    assert len(space.layers) == 1
    assert space.layers[0] is layer1


def test_to_crs_non_inplace_preserves_names():
    space = GeoSpace(crs="epsg:4326")
    layer1 = ImageLayer(
        values=np.zeros((1, 10, 10)), crs="epsg:4326", total_bounds=[0, 0, 10, 10]
    )
    space.add_layer(layer1, name="test_layer")

    new_space = space.to_crs("epsg:3857", inplace=False)
    assert new_space is not space
    new_layer = new_space.get_layer("test_layer")
    assert new_layer is not layer1
    assert new_layer.crs.to_string() == "EPSG:3857"


def test_to_crs_inplace_preserves_names():
    space = GeoSpace(crs="epsg:4326")
    layer1 = ImageLayer(
        values=np.zeros((1, 10, 10)), crs="epsg:4326", total_bounds=[0, 0, 10, 10]
    )
    space.add_layer(layer1, name="test_layer")

    space.to_crs("epsg:3857", inplace=True)
    layer = space.get_layer("test_layer")
    assert layer is layer1
    assert layer.crs.to_string() == "EPSG:3857"


def test_name_for_layer_reverse_lookup():
    space = GeoSpace()
    layer1 = ImageLayer(
        values=np.zeros((1, 10, 10)), crs="epsg:4326", total_bounds=[0, 0, 10, 10]
    )
    space.add_layer(layer1, name="test_layer")
    assert space._name_for_layer(layer1) == "test_layer"
    assert layer1.name == "test_layer"


def test_geodataframe_with_name_column():
    space = GeoSpace(crs="epsg:4326")
    gdf = gpd.GeoDataFrame(
        {"name": ["alpha", "beta"], "geometry": [Point(0, 0), Point(1, 1)]},
        crs="epsg:4326",
    )
    space.add_layer(gdf, name="regions")
    assert space.get_layer("regions") is gdf
    assert space._name_for_layer(gdf) == "regions"
    assert list(gdf["name"]) == ["alpha", "beta"]


def test_subclass_from_file_old_init_signature(tmp_path):
    class CustomRasterLayer(RasterLayer):
        def __init__(self, width, height, crs, total_bounds, model, cell_cls=Cell):
            super().__init__(width, height, crs, total_bounds, model, cell_cls)

    file_path = str(tmp_path / "test.tif")
    data = np.ones((1, 4, 4), dtype=np.float32)
    transform = rio.transform.from_bounds(0, 0, 4, 4, 4, 4)
    with rio.open(
        file_path,
        "w",
        driver="GTiff",
        height=4,
        width=4,
        count=1,
        dtype=data.dtype,
        crs="epsg:4326",
        transform=transform,
    ) as dst:
        dst.write(data)

    model = mesa.Model()
    layer = CustomRasterLayer.from_file(file_path, model=model, name="custom")
    assert isinstance(layer, CustomRasterLayer)
    assert layer.name == "custom"


def test_from_file_no_name_and_name_resolution(tmp_path):
    file_path = str(tmp_path / "test_no_name.tif")
    data = np.ones((1, 4, 4), dtype=np.float32)
    transform = rio.transform.from_bounds(0, 0, 4, 4, 4, 4)
    with rio.open(
        file_path,
        "w",
        driver="GTiff",
        height=4,
        width=4,
        count=1,
        dtype=data.dtype,
        crs="epsg:4326",
        transform=transform,
    ) as dst:
        dst.write(data)

    model = mesa.Model()
    space = GeoSpace(crs="epsg:4326")
    layer = RasterLayer.from_file(file_path, model=model)
    assert layer.name is None

    space.add_layer(layer, name="x")
    assert space.get_layer("x") is layer
    assert space._name_for_layer(layer) == "x"
    assert layer.name == "x"

    # Manually set layer name fallback
    layer2 = RasterLayer(4, 4, crs="epsg:4326", total_bounds=[0, 0, 4, 4], model=model)
    layer2.name = "manual_name"
    space.add_layer(layer2)
    assert space._name_for_layer(layer2) == "manual_name"


def test_readd_same_layer_object_under_second_name_rejected():
    space = GeoSpace()
    layer = ImageLayer(
        values=np.zeros((1, 10, 10)), crs="epsg:4326", total_bounds=[0, 0, 10, 10]
    )
    space.add_layer(layer, name="first")
    with pytest.raises(
        ValueError, match="Layer is already registered with name 'first'"
    ):
        space.add_layer(layer, name="second")
