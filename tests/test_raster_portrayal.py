"""Tests for the vectorized raster portrayal path.

Covers pixel-exact correctness, vmin/vmax handling, NaN transparency,
callable/bare-style dispatch, band z-order, multi-layer selective styling,
and legacy-path byte-identity.
"""

import base64
import warnings
from io import BytesIO

import geopandas as gpd
import ipyleaflet
import matplotlib
import matplotlib.colors
import mesa
import numpy as np
import pytest
import solara
import xyzservices.providers as xyz
from mesa.visualization.components import PropertyLayerStyle
from mesa.visualization.solara_viz import SolaraViz
from PIL import Image
from shapely.geometry import Point, Polygon

import mesa_geo as mg
import mesa_geo.visualization as mgv
import mesa_geo.visualization.components.geospace_component as gc
from mesa_geo.raster_layers import ImageLayer, RasterLayer
from mesa_geo.visualization import make_geospace_component
from mesa_geo.visualization.components.geospace_component import MapModule

# ---- Helpers ----


@pytest.fixture(autouse=True)
def _reset_agent_portrayal_warned():
    """Keep the module-level warn-once flag from leaking across tests."""
    gc._AGENT_PORTRAYAL_STATE["warned"] = False
    yield
    gc._AGENT_PORTRAYAL_STATE["warned"] = False


def _make_model_with_raster(data, *, crs="epsg:4326", band_name="band0"):
    """Create a Model+GeoSpace+RasterLayer with a single band, CRS 4326 so
    to_crs('epsg:4326') is an identity transform."""
    model = mesa.Model()
    model.space = mg.GeoSpace(crs=crs, warn_crs_conversion=False)
    h, w = data.shape
    layer = RasterLayer(w, h, crs=crs, total_bounds=[0, 0, 1, 1], model=model)
    layer.set_band(band_name, data)
    model.space.add_layer(layer, name="test_layer")
    return model, layer


def _decode_data_url_to_rgba(data_url):
    """Decode a data:image/png;base64,... URL into a (H, W, 4) uint8 array."""
    _, b64 = data_url.split(",", 1)
    raw = base64.b64decode(b64)
    img = Image.open(BytesIO(raw)).convert("RGBA")
    return np.array(img)


def _render_rasters(model, raster_portrayal):
    """Shortcut: render via MapModule and return the rasters list."""
    mm = MapModule(
        portrayal_method=None,
        tiles=xyz.OpenStreetMap.Mapnik,
        raster_portrayal=raster_portrayal,
    )
    result = mm.render(model)
    return result["layers"]["rasters"]


# ---- Tests ----


class TestPixelExactColormap:
    """Asymmetric raster, colormap="viridis", vmin=0, vmax=100."""

    def test_viridis_pixel_exact(self):
        # Asymmetric: 3 rows by 5 cols, distinct values
        data = np.array(
            [
                [0, 25, 50, 75, 100],
                [10, 30, 60, 80, 90],
                [5, 15, 45, 65, 95],
            ],
            dtype=float,
        )

        model, _ = _make_model_with_raster(data, band_name="elev")
        style = PropertyLayerStyle(colormap="viridis", vmin=0, vmax=100, alpha=1.0)
        rasters = _render_rasters(model, lambda ln, bn: style)

        assert len(rasters) == 1
        decoded = _decode_data_url_to_rgba(rasters[0]["url"])

        # Compute expected via the same matplotlib pipeline
        norm = matplotlib.colors.Normalize(vmin=0, vmax=100)
        cmap = matplotlib.colormaps["viridis"]
        expected = (cmap(norm(data)) * 255).astype(np.uint8)

        # The CRS is epsg:4326 -> epsg:4326, so the transform is identity.
        np.testing.assert_array_equal(decoded, expected)

    def test_bare_style_default_alpha(self):
        """PropertyLayerStyle defaults alpha=0.8, rendered output should reflect it."""
        data = np.array([[10, 20], [30, 40]], dtype=float)
        model, _ = _make_model_with_raster(data, band_name="elev")
        style = PropertyLayerStyle(colormap="viridis")
        rasters = _render_rasters(model, style)

        assert len(rasters) == 1
        decoded = _decode_data_url_to_rgba(rasters[0]["url"])
        assert decoded[0, 0, 3] == int(0.8 * 255)


class TestVminZeroRespected:
    """vmin=0 is honoured rather than being treated as unset."""

    def test_vmin_zero_not_autoranged(self):
        data = np.array([[50, 100], [75, 25]], dtype=float)
        model, _ = _make_model_with_raster(data, band_name="b")

        style_explicit = PropertyLayerStyle(colormap="viridis", vmin=0, vmax=100)
        style_auto = PropertyLayerStyle(colormap="viridis", vmin=None, vmax=None)

        rasters_explicit = _render_rasters(model, lambda ln, bn: style_explicit)
        rasters_auto = _render_rasters(model, lambda ln, bn: style_auto)

        px_explicit = _decode_data_url_to_rgba(rasters_explicit[0]["url"])
        px_auto = _decode_data_url_to_rgba(rasters_auto[0]["url"])

        # With vmin=0, the 25 pixel should map differently than with
        # vmin=25 (auto-ranged). The images should differ.
        assert not np.array_equal(px_explicit, px_auto)


class TestAutoRange:
    """vmin/vmax of None auto-range with nan-aware min/max."""

    def test_auto_range_nan_aware(self):
        data = np.array([[np.nan, 20], [40, 60]], dtype=float)
        model, _ = _make_model_with_raster(data, band_name="b")

        style = PropertyLayerStyle(colormap="viridis", vmin=None, vmax=None, alpha=1.0)
        rasters = _render_rasters(model, lambda ln, bn: style)

        assert len(rasters) == 1
        decoded = _decode_data_url_to_rgba(rasters[0]["url"])

        # Auto-range should be nanmin=20, nanmax=60
        norm = matplotlib.colors.Normalize(vmin=20, vmax=60)
        cmap = matplotlib.colormaps["viridis"]
        expected = (cmap(norm(data)) * 255).astype(np.uint8)
        # NaN pixel should be alpha=0
        expected[0, 0, 3] = 0
        np.testing.assert_array_equal(decoded, expected)


class TestUniformColor:
    """A uniform color= style produces a constant RGBA."""

    def test_uniform_red(self):
        data = np.array([[50, 100], [0.1, 75]], dtype=float)
        model, _ = _make_model_with_raster(data, band_name="b")

        style = PropertyLayerStyle(color="red", alpha=1.0, vmin=0, vmax=100)
        rasters = _render_rasters(model, lambda ln, bn: style)

        assert len(rasters) == 1
        decoded = _decode_data_url_to_rgba(rasters[0]["url"])

        red_rgba = matplotlib.colors.to_rgba("red")
        # RGB channels should all be (255, 0, 0)
        assert np.all(decoded[..., 0] == int(red_rgba[0] * 255))
        assert np.all(decoded[..., 1] == int(red_rgba[1] * 255))
        assert np.all(decoded[..., 2] == int(red_rgba[2] * 255))
        # Alpha should be proportional to normalized data value
        # data=100 -> norm=1.0 -> alpha=255; data=0.1 -> norm~0.001 -> alpha~0
        assert decoded[0, 1, 3] == 255  # data=100, fully opaque


class TestNanTransparency:
    """A NaN cell is fully transparent."""

    def test_nan_alpha_zero(self):
        data = np.array([[np.nan, 50], [25, 75]], dtype=float)
        model, _ = _make_model_with_raster(data, band_name="b")

        style = PropertyLayerStyle(colormap="viridis", vmin=0, vmax=100)
        rasters = _render_rasters(model, lambda ln, bn: style)

        decoded = _decode_data_url_to_rgba(rasters[0]["url"])
        assert decoded[0, 0, 3] == 0  # NaN pixel is fully transparent


class TestCallableReturnsNone:
    """A callable returning None for a band produces no overlay."""

    def test_none_skips_band(self):
        data = np.array([[10, 20], [30, 40]], dtype=float)
        model, layer = _make_model_with_raster(data, band_name="visible")
        layer.set_band("hidden", np.ones((2, 2)))

        def selective(layer_name, band_name):
            if band_name == "hidden":
                return None
            return PropertyLayerStyle(colormap="viridis", vmin=0, vmax=100)

        rasters = _render_rasters(model, selective)
        # Only 'visible' should produce an overlay
        assert len(rasters) == 1


class TestBareStyleAllBands:
    """A bare style object applies to every band."""

    def test_bare_style_covers_all(self):
        data1 = np.array([[10, 20], [30, 40]], dtype=float)
        data2 = np.array([[50, 60], [70, 80]], dtype=float)
        model, layer = _make_model_with_raster(data1, band_name="a")
        layer.set_band("b", data2)

        style = PropertyLayerStyle(colormap="viridis", vmin=0, vmax=100)
        rasters = _render_rasters(model, style)
        assert len(rasters) == 2


class TestSelectiveMultiLayer:
    """With two raster layers, only the styled one is drawn."""

    def test_two_layers_one_styled(self):
        model = mesa.Model()
        model.space = mg.GeoSpace(crs="epsg:4326", warn_crs_conversion=False)

        layer1 = RasterLayer(
            2, 2, crs="epsg:4326", total_bounds=[0, 0, 1, 1], model=model
        )
        layer1.set_band("elev", np.array([[10, 20], [30, 40]], dtype=float))
        model.space.add_layer(layer1, name="styled")

        layer2 = RasterLayer(
            2, 2, crs="epsg:4326", total_bounds=[0, 0, 1, 1], model=model
        )
        layer2.set_band("temp", np.array([[5, 6], [7, 8]], dtype=float))
        model.space.add_layer(layer2, name="unstyled")

        def portrayal(layer_name, band_name):
            if layer_name == "styled":
                return PropertyLayerStyle(colormap="viridis", vmin=0, vmax=100)
            return None

        rasters = _render_rasters(model, portrayal)
        # Only the "styled" layer produced an overlay
        assert len(rasters) == 1


class TestLegacyPathUntouched:
    """raster_portrayal=None keeps the original to_image path."""

    def test_legacy_renders(self):
        """When raster_portrayal is None the old to_image codepath runs."""
        model = mesa.Model()
        model.space = mg.GeoSpace(crs="epsg:4326", warn_crs_conversion=False)
        layer = RasterLayer(
            2, 2, crs="epsg:4326", total_bounds=[0, 0, 1, 1], model=model
        )
        layer.apply_raster(np.array([[[10, 20], [30, 40]]]))
        model.space.add_layer(layer)

        def old_portrayal(cell):
            # apply_raster with no attr_name generates "attribute_0"
            v = cell.attribute_0
            return (min(v * 2, 255), v, 0, 255)

        mm_legacy = MapModule(
            portrayal_method=old_portrayal, tiles=xyz.OpenStreetMap.Mapnik
        )
        result = mm_legacy.render(model)
        rasters = result["layers"]["rasters"]
        assert len(rasters) == 1
        assert rasters[0]["url"].startswith("data:image/png;base64,")


class TestBandZOrder:
    """Band overlays follow _data insertion order."""

    def test_overlay_order_matches_insertion(self):
        """Bands added as a, b, c must produce overlays in that order."""
        model = mesa.Model()
        model.space = mg.GeoSpace(crs="epsg:4326", warn_crs_conversion=False)
        layer = RasterLayer(
            2, 2, crs="epsg:4326", total_bounds=[0, 0, 1, 1], model=model
        )
        layer.set_band("alpha_band", np.array([[10, 20], [30, 40]], dtype=float))
        layer.set_band("beta_band", np.array([[50, 60], [70, 80]], dtype=float))
        model.space.add_layer(layer, name="ordered")

        call_order = []

        def tracking_portrayal(layer_name, band_name):
            call_order.append(band_name)
            return PropertyLayerStyle(colormap="viridis", vmin=0, vmax=100)

        _render_rasters(model, tracking_portrayal)
        assert call_order == ["alpha_band", "beta_band"]


class TestAllNanBandSkipped:
    """An all-NaN band is skipped, without warning or overlay."""

    def test_all_nan_no_overlay(self):
        data = np.full((3, 3), np.nan)
        model, _ = _make_model_with_raster(data, band_name="nan_band")

        style = PropertyLayerStyle(colormap="viridis")
        rasters = _render_rasters(model, lambda ln, bn: style)
        assert len(rasters) == 0


class TestConstantBandRendered:
    """A constant band is drawn flat rather than skipped."""

    def test_constant_value_renders_flat_color(self):
        data = np.full((3, 3), 42.0)
        model, _ = _make_model_with_raster(data, band_name="const_band")

        style = PropertyLayerStyle(colormap="viridis", alpha=1.0)
        rasters = _render_rasters(model, lambda ln, bn: style)
        assert len(rasters) == 1
        decoded = _decode_data_url_to_rgba(rasters[0]["url"])

        # Verify it's a uniform flat color and not transparent (alpha=255)
        assert decoded[0, 0, 3] == 255

        # Since vmin==vmax, Normalize maps it to 0 (low end of colormap)
        norm = matplotlib.colors.Normalize(vmin=42.0, vmax=42.0)
        cmap = matplotlib.colormaps["viridis"]
        expected = (cmap(norm(data)) * 255).astype(np.uint8)
        np.testing.assert_array_equal(decoded, expected)

    def test_constant_value_renders_color_path(self):
        data = np.full((3, 3), 42.0)
        model, _ = _make_model_with_raster(data, band_name="const_color_band")

        style = PropertyLayerStyle(color="red", alpha=1.0)
        rasters = _render_rasters(model, lambda ln, bn: style)
        assert len(rasters) == 1
        decoded = _decode_data_url_to_rgba(rasters[0]["url"])

        # In the color path with vmin == vmax, the band must render visibly
        # with full alpha rather than being transparent (alpha=255).
        red_rgba = matplotlib.colors.to_rgba("red")
        assert np.all(decoded[..., 0] == int(red_rgba[0] * 255))
        assert np.all(decoded[..., 1] == int(red_rgba[1] * 255))
        assert np.all(decoded[..., 2] == int(red_rgba[2] * 255))
        assert np.all(decoded[..., 3] == 255)


class TestPortrayalValidation:
    """Non-callable / invalid raster_portrayal raises TypeError."""

    def test_dict_portrayal_raises_type_error(self):
        data = np.array([[10, 20], [30, 40]], dtype=float)
        model, _ = _make_model_with_raster(data, band_name="a")

        with pytest.raises(
            TypeError,
            match=r"'raster_portrayal' must be a callable.*or a PropertyLayerStyle instance",
        ):
            _render_rasters(model, {"colormap": "viridis"})

    def test_callable_returning_dict_raises_type_error(self):
        data = np.array([[10, 20], [30, 40]], dtype=float)
        model, _ = _make_model_with_raster(data, band_name="a")

        with pytest.raises(
            TypeError,
            match=r"Portrayal for band 'a' .* must be a PropertyLayerStyle or None",
        ):
            _render_rasters(model, lambda ln, bn: {"colormap": "viridis"})

    def test_property_layer_style_mutually_exclusive(self):
        with pytest.raises(
            ValueError, match="Specify either 'color' or 'colormap', not both"
        ):
            PropertyLayerStyle(color="red", colormap="viridis")

    def test_property_layer_style_requires_at_least_one(self):
        with pytest.raises(ValueError, match="Specify one of 'color' or 'colormap'"):
            PropertyLayerStyle()


class TestAgentPortrayalNoneGuard:
    """When both agent_portrayal and raster_portrayal are None and a RasterLayer exists, raise ValueError."""

    def test_agent_portrayal_none_raises_clear_error(self):
        data = np.array([[10, 20], [30, 40]], dtype=float)
        model, _ = _make_model_with_raster(data, band_name="a")

        mm = MapModule(
            portrayal_method=None,
            tiles=xyz.OpenStreetMap.Mapnik,
            raster_portrayal=None,
        )
        with pytest.raises(
            ValueError,
            match=r"neither 'raster_portrayal' nor 'agent_portrayal' was provided",
        ):
            mm.render(model)


class TestRenderByteIdentitySnapshot:
    """Byte-identity snapshot test for rendering refactor.

    Ensures that internal separation into _RasterRenderer and _VectorRenderer
    produces bit-for-bit identical outputs for both legacy and new portrayal paths.
    """

    @pytest.fixture
    def fixture_model(self):
        model = mesa.Model()
        model.space = mg.GeoSpace(crs="epsg:4326", warn_crs_conversion=False)

        # 1. Raster layer
        rl = mg.RasterLayer(
            2, 2, crs="epsg:4326", total_bounds=[0, 0, 2, 2], model=model
        )
        rl.apply_raster(np.array([[[10, 20], [30, 40]]]))
        model.space.add_layer(rl, name="grid")

        # 2. Vector GeoDataFrame layer
        gdf = gpd.GeoDataFrame(
            geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 0)])],
            crs="epsg:4326",
        )
        model.space.add_layer(gdf)

        # 3. Agents
        creator = mg.AgentCreator(agent_class=mg.GeoAgent, model=model, crs="epsg:4326")
        pa = creator.create_agent(Point(0.5, 0.5))
        pya = creator.create_agent(Polygon([(0, 0), (2, 0), (2, 2), (0, 0)]))
        model.space.add_agents([pa, pya])

        return model

    def test_legacy_path_snapshot(self, fixture_model):
        def legacy_p(agent_or_cell):
            if isinstance(agent_or_cell, mg.Cell):
                v = agent_or_cell.attribute_0
                return (v, v, v, 255)
            if isinstance(agent_or_cell.geometry, Point):
                return {
                    "marker_type": "Circle",
                    "radius": 10,
                    "color": "red",
                    "description": "point",
                }
            return {"color": "blue", "weight": 2, "description": "poly"}

        mm = MapModule(portrayal_method=legacy_p, tiles=xyz.OpenStreetMap.Mapnik)
        out = mm.render(fixture_model)

        decoded = _decode_data_url_to_rgba(out["layers"]["rasters"][0]["url"])
        # Legacy to_image+write_png normalises float data by per-channel max:
        # 10/40*255 ≈ 63, 20/40*255 ≈ 127, etc. The new portrayal path avoids this.
        expected_rgba = np.array(
            [
                [[63, 63, 63, 255], [127, 127, 127, 255]],
                [[191, 191, 191, 255], [255, 255, 255, 255]],
            ],
            dtype=np.uint8,
        )
        np.testing.assert_array_equal(decoded, expected_rgba)
        assert out["layers"]["rasters"][0]["bounds"] == [[0.0, 0.0], [2.0, 2.0]]
        assert out["layers"]["total_bounds"] == [[0.0, 0.0], [2.0, 2.0]]
        assert len(out["layers"]["vectors"]) == 1
        assert len(out["agents"][0]["features"]) == 1
        assert out["agents"][0]["features"][0]["properties"]["style"] == {
            "color": "#0000ff",
            "weight": 2,
        }
        assert len(out["agents"][1]) == 1
        assert out["agents"][1][0].location == [0.5, 0.5]
        assert out["agents"][1][0].radius == 10
        assert out["agents"][1][0].color == "#ff0000"

    def test_new_portrayal_path_snapshot(self, fixture_model):
        def agent_p(a):
            if isinstance(a.geometry, Point):
                return {
                    "marker_type": "Circle",
                    "radius": 10,
                    "color": "red",
                    "description": "point",
                }
            return {"color": "blue", "weight": 2, "description": "poly"}

        mm = MapModule(
            portrayal_method=agent_p,
            tiles=xyz.OpenStreetMap.Mapnik,
            raster_portrayal=PropertyLayerStyle(colormap="viridis"),
        )
        out = mm.render(fixture_model)

        decoded = _decode_data_url_to_rgba(out["layers"]["rasters"][0]["url"])
        data = np.array([[10, 20], [30, 40]], dtype=float)
        norm = matplotlib.colors.Normalize(vmin=10, vmax=40)
        cmap = matplotlib.colormaps["viridis"]
        expected_rgba = (cmap(norm(data)) * 255).astype(np.uint8)
        expected_rgba[..., 3] = int(0.8 * 255)  # default alpha is 0.8
        np.testing.assert_array_equal(decoded, expected_rgba)
        assert out["layers"]["rasters"][0]["bounds"] == [[0.0, 0.0], [2.0, 2.0]]
        assert out["layers"]["total_bounds"] == [[0.0, 0.0], [2.0, 2.0]]
        assert len(out["layers"]["vectors"]) == 1
        assert len(out["agents"][0]["features"]) == 1
        assert out["agents"][0]["features"][0]["properties"]["style"] == {
            "color": "#0000ff",
            "weight": 2,
        }
        assert len(out["agents"][1]) == 1
        assert out["agents"][1][0].location == [0.5, 0.5]
        assert out["agents"][1][0].radius == 10
        assert out["agents"][1][0].color == "#ff0000"


class TestLiveCellAttributeMutation:
    """Mutating cell attributes between render calls must be reflected in output."""

    def test_cell_attribute_mutation_reflected(self):
        data = np.array([[10.0, 20.0], [30.0, 40.0]])
        model, layer = _make_model_with_raster(data, band_name="elevation")
        style = PropertyLayerStyle(colormap="viridis", vmin=0, vmax=100)
        mm = MapModule(
            portrayal_method=lambda _: {},
            tiles=xyz.OpenStreetMap.Mapnik,
            raster_portrayal=style,
        )

        out1 = mm.render(model)
        decoded1 = _decode_data_url_to_rgba(out1["layers"]["rasters"][0]["url"])

        # Mutate cell attribute directly
        layer.cells[0][1].elevation = 99.0

        out2 = mm.render(model)
        decoded2 = _decode_data_url_to_rgba(out2["layers"]["rasters"][0]["url"])

        assert not np.array_equal(decoded1, decoded2)
        cmap = matplotlib.colormaps["viridis"]
        norm = matplotlib.colors.Normalize(vmin=0, vmax=100)
        expected_pixel = (np.array(cmap(norm(99.0))) * 255).astype(np.uint8)
        expected_pixel[3] = int(0.8 * 255)
        np.testing.assert_array_equal(decoded2[0, 0], expected_pixel)


class TestCustomAndListColormaps:
    """Support list of colors, RGB sequences, and unregistered Colormap objects."""

    def test_list_of_colors_raster(self):
        data = np.array([[10.0, 20.0], [30.0, 40.0]])
        model, _ = _make_model_with_raster(data, band_name="val")
        style = PropertyLayerStyle(colormap=["#ff0000", "#0000ff"], vmin=0, vmax=100)
        mm = MapModule(
            portrayal_method=lambda _: {},
            tiles=xyz.OpenStreetMap.Mapnik,
            raster_portrayal=style,
        )
        out = mm.render(model)
        assert len(out["layers"]["rasters"]) == 1

    def test_list_of_rgb_sequences_raster(self):
        data = np.array([[10.0, 20.0], [30.0, 40.0]])
        model, _ = _make_model_with_raster(data, band_name="val")
        style = PropertyLayerStyle(
            colormap=[[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], vmin=0, vmax=100
        )
        mm = MapModule(
            portrayal_method=lambda _: {},
            tiles=xyz.OpenStreetMap.Mapnik,
            raster_portrayal=style,
        )
        out = mm.render(model)
        assert len(out["layers"]["rasters"]) == 1

    def test_unregistered_colormap_object(self):
        data = np.array([[10.0, 20.0], [30.0, 40.0]])
        model, _ = _make_model_with_raster(data, band_name="val")
        custom_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
            "unregistered_test_cmap", ["green", "yellow"]
        )
        style = PropertyLayerStyle(colormap=custom_cmap, vmin=0, vmax=100)
        mm = MapModule(
            portrayal_method=lambda _: {},
            tiles=xyz.OpenStreetMap.Mapnik,
            raster_portrayal=style,
        )
        out = mm.render(model)
        assert len(out["layers"]["rasters"]) == 1


class TestPortrayalDictNotMutated:
    """Portrayal dict returned by agent_portrayal must not be mutated in-place."""

    def test_portrayal_dict_not_mutated_across_agents(self):
        model = mesa.Model()
        model.space = mg.GeoSpace(crs="epsg:4326")
        creator = mg.AgentCreator(agent_class=mg.GeoAgent, model=model, crs="epsg:4326")
        agent1 = creator.create_agent(Point(1.0, 2.0))
        agent2 = creator.create_agent(Point(3.0, 4.0))
        model.space.add_agents([agent1, agent2])

        shared_dict = {
            "marker_type": "CircleMarker",
            "radius": 15,
            "color": "red",
            "fillColor": "blue",
            "description": "shared_popup",
        }

        def agent_p(_):
            return shared_dict

        mm = MapModule(portrayal_method=agent_p, tiles=xyz.OpenStreetMap.Mapnik)
        out = mm.render(model)

        assert shared_dict["marker_type"] == "CircleMarker"
        assert shared_dict["description"] == "shared_popup"
        assert shared_dict["radius"] == 15
        assert shared_dict["fillColor"] == "blue"

        markers = out["agents"][1]
        assert len(markers) == 2
        assert isinstance(markers[0], ipyleaflet.CircleMarker)
        assert isinstance(markers[1], ipyleaflet.CircleMarker)
        assert markers[0].radius == markers[1].radius == 15
        assert markers[0].color == markers[1].color == "#ff0000"
        assert markers[0].fill_color == markers[1].fill_color == "#0000ff"


class TestVectorColorNormalization:
    """Test matplotlib color normalization to CSS hex strings in _VectorRenderer."""

    def test_css_color_utility_directly(self):
        css = gc._VectorRenderer._css_color
        assert css("tab:blue") == "#1f77b4"
        assert css("C0") == "#1f77b4"
        assert css("C1") == "#ff7f0e"
        assert css("xkcd:sky blue") == "#75bbfd"
        assert css("red") == "#ff0000"
        assert css("#123456") == "#123456"
        assert css((1.0, 0.0, 0.0)) == "#ff0000"
        assert css((0.0, 1.0, 0.0, 0.5)) == "#00ff0080"
        assert css("#ff000080") == "#ff000080"
        assert css(None) is None
        assert css("transparent") == "transparent"

    def test_point_marker_color_and_fillcolor_normalized(self):
        model = mesa.Model()
        model.space = mg.GeoSpace(crs="epsg:4326")
        creator = mg.AgentCreator(agent_class=mg.GeoAgent, model=model, crs="epsg:4326")
        agent = creator.create_agent(Point(1.0, 2.0))
        model.space.add_agents([agent])

        def agent_p(_):
            return {
                "marker_type": "Circle",
                "color": "tab:blue",
                "fillColor": "C0",
            }

        mm = MapModule(portrayal_method=agent_p, tiles=xyz.OpenStreetMap.Mapnik)
        out = mm.render(model)
        marker = out["agents"][1][0]
        assert marker.color == "#1f77b4"
        assert marker.fill_color == "#1f77b4"

    def test_circlemarker_color_normalized(self):
        model = mesa.Model()
        model.space = mg.GeoSpace(crs="epsg:4326")
        creator = mg.AgentCreator(agent_class=mg.GeoAgent, model=model, crs="epsg:4326")
        agent = creator.create_agent(Point(1.0, 2.0))
        model.space.add_agents([agent])

        def agent_p(_):
            return {
                "marker_type": "CircleMarker",
                "color": "C2",
                "fillColor": "red",
            }

        mm = MapModule(portrayal_method=agent_p, tiles=xyz.OpenStreetMap.Mapnik)
        out = mm.render(model)
        marker = out["agents"][1][0]
        assert marker.color == "#2ca02c"
        assert marker.fill_color == "#ff0000"

    def test_polygon_color_and_fillcolor_normalized(self):
        model = mesa.Model()
        model.space = mg.GeoSpace(crs="epsg:4326")
        creator = mg.AgentCreator(agent_class=mg.GeoAgent, model=model, crs="epsg:4326")
        poly_agent = creator.create_agent(Polygon([(0, 0), (1, 0), (1, 1), (0, 0)]))
        model.space.add_agents([poly_agent])

        def agent_p(_):
            return {
                "color": "xkcd:sky blue",
                "fillColor": "tab:orange",
                "weight": 3,
            }

        mm = MapModule(portrayal_method=agent_p, tiles=xyz.OpenStreetMap.Mapnik)
        out = mm.render(model)
        style = out["agents"][0]["features"][0]["properties"]["style"]
        assert style["color"] == "#75bbfd"
        assert style["fillColor"] == "#ff7f0e"
        assert style["weight"] == 3

    def test_none_and_css_keyword_preserved(self):
        model = mesa.Model()
        model.space = mg.GeoSpace(crs="epsg:4326")
        creator = mg.AgentCreator(agent_class=mg.GeoAgent, model=model, crs="epsg:4326")
        poly_agent = creator.create_agent(Polygon([(0, 0), (1, 0), (1, 1), (0, 0)]))
        model.space.add_agents([poly_agent])

        def agent_p(_):
            return {
                "color": None,
                "fillColor": "transparent",
            }

        mm = MapModule(portrayal_method=agent_p, tiles=xyz.OpenStreetMap.Mapnik)
        out = mm.render(model)
        style = out["agents"][0]["features"][0]["properties"]["style"]
        assert style["color"] is None
        assert style["fillColor"] == "transparent"


class TestColormapResolution:
    """Colormap values are resolved to a Colormap where the style is validated."""

    def test_tuple_renders_identically_to_list(self):
        data = np.array([[10.0, 20.0], [30.0, 40.0]])
        model_list, _ = _make_model_with_raster(data, band_name="val")
        model_tuple, _ = _make_model_with_raster(data, band_name="val")

        as_list = _render_rasters(
            model_list,
            PropertyLayerStyle(colormap=["#ff0000", "#0000ff"], vmin=0, vmax=100),
        )
        as_tuple = _render_rasters(
            model_tuple,
            PropertyLayerStyle(colormap=("#ff0000", "#0000ff"), vmin=0, vmax=100),
        )
        assert as_list[0]["url"] == as_tuple[0]["url"]

    @pytest.mark.parametrize("colormap", [[], (), ""])
    def test_empty_colormap_raises_type_error(self, colormap):
        data = np.array([[10.0, 20.0], [30.0, 40.0]])
        model, _ = _make_model_with_raster(data, band_name="val")
        style = PropertyLayerStyle(colormap=colormap, vmin=0, vmax=100)

        with pytest.raises(TypeError, match="non-empty sequence of colors"):
            _render_rasters(model, style)

    @pytest.mark.parametrize("colormap", [5, {"a": 1}])
    def test_unrecognised_colormap_raises_type_error(self, colormap):
        data = np.array([[10.0, 20.0], [30.0, 40.0]])
        model, _ = _make_model_with_raster(data, band_name="val")
        style = PropertyLayerStyle(colormap=colormap, vmin=0, vmax=100)

        with pytest.raises(TypeError, match="Invalid 'colormap' for band 'val'"):
            _render_rasters(model, style)

    def test_unregistered_colormap_instance_renders(self):
        data = np.array([[10.0, 20.0], [30.0, 40.0]])
        model, _ = _make_model_with_raster(data, band_name="val")
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
            "unregistered", ["red", "blue"]
        )

        rasters = _render_rasters(
            model, PropertyLayerStyle(colormap=cmap, vmin=0, vmax=100)
        )
        decoded = _decode_data_url_to_rgba(rasters[0]["url"])
        expected = (np.array(cmap(0.1)) * 255).astype(np.uint8)
        assert decoded[0, 0, 0] == expected[0]
        assert decoded[0, 0, 2] == expected[2]


class TestVminVmaxOrdering:
    """Inverted normalization bounds fail loudly instead of blanking the map."""

    def test_vmin_greater_than_vmax_raises(self):
        data = np.array([[10.0, 20.0], [30.0, 40.0]])
        model, _ = _make_model_with_raster(data, band_name="val")
        style = PropertyLayerStyle(colormap="viridis", vmin=10, vmax=0)

        with pytest.raises(ValueError, match=r"'vmin' \(10\).*'vmax' \(0\)"):
            _render_rasters(model, style)

    def test_equal_bounds_still_render(self):
        data = np.array([[10.0, 10.0], [10.0, 10.0]])
        model, _ = _make_model_with_raster(data, band_name="val")

        rasters = _render_rasters(
            model, PropertyLayerStyle(colormap="viridis", vmin=10, vmax=10)
        )
        assert len(rasters) == 1


class TestMissingAgentPortrayalWarns:
    """A space with agents and no agent_portrayal warns once, not silently."""

    @staticmethod
    def _model_with_agent():
        model = mesa.Model()
        model.space = mg.GeoSpace(crs="epsg:4326")
        creator = mg.AgentCreator(agent_class=mg.GeoAgent, model=model, crs="epsg:4326")
        model.space.add_agents([creator.create_agent(Point(1.0, 2.0))])
        return model

    def test_warns_when_agents_present_and_no_portrayal(self):
        model = self._model_with_agent()
        mm = MapModule(portrayal_method=None, tiles=xyz.OpenStreetMap.Mapnik)

        with pytest.warns(UserWarning, match="no 'agent_portrayal' was provided"):
            mm.render(model)

    def test_warns_once_across_renders(self):
        model = self._model_with_agent()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for _ in range(3):
                MapModule(portrayal_method=None, tiles=xyz.OpenStreetMap.Mapnik).render(
                    model
                )
            relevant = [w for w in caught if "agent_portrayal" in str(w.message)]
        assert len(relevant) == 1

    def test_no_warning_when_space_has_no_agents(self):
        data = np.array([[10.0, 20.0], [30.0, 40.0]])
        model, _ = _make_model_with_raster(data, band_name="val")

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _render_rasters(model, PropertyLayerStyle(colormap="viridis"))
            relevant = [w for w in caught if "agent_portrayal" in str(w.message)]
        assert relevant == []

    def test_no_warning_when_portrayal_given(self):
        model = self._model_with_agent()
        mm = MapModule(
            portrayal_method=lambda _: {"color": "red"},
            tiles=xyz.OpenStreetMap.Mapnik,
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            mm.render(model)
            relevant = [w for w in caught if "agent_portrayal" in str(w.message)]
        assert relevant == []


class TestMarkerAlphaRouting:
    """Marker alpha travels as opacity; the GeoJSON path keeps 8-digit hex."""

    @staticmethod
    def _render_marker(portrayal):
        model = mesa.Model()
        model.space = mg.GeoSpace(crs="epsg:4326")
        creator = mg.AgentCreator(agent_class=mg.GeoAgent, model=model, crs="epsg:4326")
        model.space.add_agents([creator.create_agent(Point(1.0, 2.0))])
        mm = MapModule(
            portrayal_method=lambda _: portrayal, tiles=xyz.OpenStreetMap.Mapnik
        )
        return mm.render(model)["agents"][1][0]

    def test_translucent_color_split_into_opacity(self):
        marker = self._render_marker(
            {
                "marker_type": "Circle",
                "color": "#ff000080",
                "fillColor": (0.0, 0.0, 1.0, 0.25),
            }
        )
        assert marker.color == "#ff0000"
        assert marker.opacity == pytest.approx(128 / 255)
        assert marker.fill_color == "#0000ff"
        assert marker.fill_opacity == pytest.approx(0.25)

    def test_opaque_color_leaves_opacity_at_default(self):
        marker = self._render_marker({"marker_type": "Circle", "color": "red"})
        assert marker.color == "#ff0000"
        assert marker.opacity == 1.0

    def test_explicit_opacity_not_overridden(self):
        marker = self._render_marker(
            {"marker_type": "Circle", "color": "#ff000080", "opacity": 0.9}
        )
        assert marker.opacity == pytest.approx(0.9)

    def test_geojson_path_keeps_eight_digit_hex(self):
        model = mesa.Model()
        model.space = mg.GeoSpace(crs="epsg:4326")
        creator = mg.AgentCreator(agent_class=mg.GeoAgent, model=model, crs="epsg:4326")
        model.space.add_agents(
            [creator.create_agent(Polygon([(0, 0), (1, 0), (1, 1), (0, 0)]))]
        )
        mm = MapModule(
            portrayal_method=lambda _: {"fillColor": (0.0, 1.0, 0.0, 0.5)},
            tiles=xyz.OpenStreetMap.Mapnik,
        )
        style = mm.render(model)["agents"][0]["features"][0]["properties"]["style"]
        assert style["fillColor"] == "#00ff0080"


class TestReprojectedRasterPixelExact:
    """The uint8 cast must happen after to_crs, which discards earlier casts."""

    @staticmethod
    def _mercator_model(data):
        model = mesa.Model()
        model.space = mg.GeoSpace(crs="epsg:3857", warn_crs_conversion=False)
        h, w = data.shape
        layer = RasterLayer(
            w,
            h,
            crs="epsg:3857",
            total_bounds=[0, 0, 1_000_000, 1_000_000],
            model=model,
        )
        layer.set_band("val", data)
        model.space.add_layer(layer, name="merc")
        return model, layer

    def test_reprojected_render_matches_cast_after_warp(self):
        data = np.array([[0.0, 25.0, 50.0], [75.0, 100.0, 10.0], [90.0, 40.0, 60.0]])
        model, layer = self._mercator_model(data)
        style = PropertyLayerStyle(colormap="viridis", vmin=0, vmax=100, alpha=0.8)

        rasters = _render_rasters(model, style)
        assert len(rasters) == 1
        decoded = _decode_data_url_to_rgba(rasters[0]["url"])

        cmap = matplotlib.colormaps["viridis"]
        norm = matplotlib.colors.Normalize(vmin=0, vmax=100)
        rgba = cmap(norm(layer.get_band("val")))
        rgba[..., 3] *= style.alpha
        warped = ImageLayer(
            values=np.clip(rgba, 0, 1).transpose([2, 0, 1]),
            crs=layer.crs,
            total_bounds=layer.total_bounds,
        ).to_crs("epsg:4326")
        expected = (np.clip(warped.values.transpose([1, 2, 0]), 0, 1) * 255).astype(
            np.uint8
        )

        np.testing.assert_array_equal(decoded, expected)

    def test_reprojection_actually_warped(self):
        """Guard the guard: a no-op transform would make the test above vacuous."""
        data = np.array([[0.0, 50.0], [100.0, 25.0]])
        model, _ = self._mercator_model(data)

        rasters = _render_rasters(
            model, PropertyLayerStyle(colormap="viridis", vmin=0, vmax=100)
        )
        bounds = rasters[0]["bounds"]
        assert bounds != [[0, 0], [1_000_000, 1_000_000]]
        assert -90 < bounds[0][0] < 90
        assert -180 < bounds[0][1] < 180


class TestMarkerOverridePoint:
    """MapModule._get_marker stays an override point after the renderer split."""

    @staticmethod
    def _model_with_point():
        model = mesa.Model()
        model.space = mg.GeoSpace(crs="epsg:4326")
        creator = mg.AgentCreator(agent_class=mg.GeoAgent, model=model, crs="epsg:4326")
        model.space.add_agents([creator.create_agent(Point(1.0, 2.0))])
        return model

    def test_subclass_override_is_called(self):
        class SentinelError(Exception):
            pass

        class OverridingMap(MapModule):
            def _get_marker(self, location, properties):
                raise SentinelError

        mm = OverridingMap(
            portrayal_method=lambda _: {"color": "red"},
            tiles=xyz.OpenStreetMap.Mapnik,
        )
        with pytest.raises(SentinelError):
            mm.render(self._model_with_point())

    def test_override_return_value_reaches_output(self):
        sentinel = object()

        class SentinelMap(MapModule):
            def _get_marker(self, location, properties):
                return sentinel

        mm = SentinelMap(
            portrayal_method=lambda _: {"color": "red"},
            tiles=xyz.OpenStreetMap.Mapnik,
        )
        assert mm.render(self._model_with_point())["agents"][1] == [sentinel]

    def test_subclass_can_extend_default(self):
        class TitledMap(MapModule):
            def _get_marker(self, location, properties):
                marker = super()._get_marker(location, properties)
                marker.title = "custom"
                return marker

        mm = TitledMap(
            portrayal_method=lambda _: {"color": "red"},
            tiles=xyz.OpenStreetMap.Mapnik,
        )
        marker = mm.render(self._model_with_point())["agents"][1][0]
        assert marker.title == "custom"
        assert marker.color == "#ff0000"


class TestCircleRadiusDefault:
    """radius=5 is defaulted only when the caller named no marker type."""

    @staticmethod
    def _render_marker(portrayal):
        model = mesa.Model()
        model.space = mg.GeoSpace(crs="epsg:4326")
        creator = mg.AgentCreator(agent_class=mg.GeoAgent, model=model, crs="epsg:4326")
        model.space.add_agents([creator.create_agent(Point(1.0, 2.0))])
        mm = MapModule(
            portrayal_method=lambda _: portrayal, tiles=xyz.OpenStreetMap.Mapnik
        )
        return mm.render(model)["agents"][1][0]

    def test_absent_marker_type_gets_radius_five(self):
        assert self._render_marker({"color": "red"}).radius == 5

    def test_explicit_circle_keeps_ipyleaflet_default(self):
        marker = self._render_marker({"marker_type": "Circle", "color": "red"})
        assert marker.radius == ipyleaflet.Circle().radius

    def test_explicit_radius_is_honoured(self):
        assert self._render_marker({"radius": 12, "color": "red"}).radius == 12
        assert (
            self._render_marker(
                {"marker_type": "Circle", "radius": 12, "color": "red"}
            ).radius
            == 12
        )


class TestNoneAndTransparentPassthrough:
    """'none' and 'transparent' are not interchangeable with #00000000 in Leaflet."""

    @pytest.mark.parametrize("keyword", ["none", "transparent", "None", " none "])
    def test_css_color_passes_keyword_through(self, keyword):
        assert gc._VectorRenderer._css_color(keyword) == keyword

    def test_geojson_path_keeps_none(self):
        model = mesa.Model()
        model.space = mg.GeoSpace(crs="epsg:4326")
        creator = mg.AgentCreator(agent_class=mg.GeoAgent, model=model, crs="epsg:4326")
        model.space.add_agents(
            [creator.create_agent(Polygon([(0, 0), (1, 0), (1, 1), (0, 0)]))]
        )
        mm = MapModule(
            portrayal_method=lambda _: {"color": "none", "fillColor": "transparent"},
            tiles=xyz.OpenStreetMap.Mapnik,
        )
        style = mm.render(model)["agents"][0]["features"][0]["properties"]["style"]
        assert style["color"] == "none"
        assert style["fillColor"] == "transparent"


class TestRasterPortrayalForwarding:
    """raster_portrayal is forwarded only when set, and does reach MapModule."""

    @staticmethod
    def _render_component(mocker, **component_kwargs):
        model = mesa.Model()
        mocker.patch.object(mesa.Model, "__new__", return_value=model)
        mocker.patch.object(mesa.Model, "__init__", return_value=None)
        solara.render(
            SolaraViz(
                model,
                components=[
                    make_geospace_component({"color": "gray"}, **component_kwargs)
                ],
            )
        )
        return model

    def test_style_reaches_geospace_leaflet_and_mapmodule(self, mocker):
        leaflet_spy = mocker.spy(mgv.components.geospace_component, "GeoSpaceLeaflet")
        mapmodule_spy = mocker.spy(mgv.components.geospace_component, "MapModule")
        style = PropertyLayerStyle(colormap="viridis")

        model = self._render_component(mocker, raster_portrayal=style)

        leaflet_spy.assert_called_with(
            model,
            {"color": "gray"},
            None,
            xyz.OpenStreetMap.Mapnik,
            raster_portrayal=style,
        )
        assert mapmodule_spy.call_args.kwargs["raster_portrayal"] is style

    def test_default_call_omits_the_keyword(self, mocker):
        leaflet_spy = mocker.spy(mgv.components.geospace_component, "GeoSpaceLeaflet")

        model = self._render_component(mocker)

        leaflet_spy.assert_called_with(
            model, {"color": "gray"}, None, xyz.OpenStreetMap.Mapnik
        )
        assert "raster_portrayal" not in leaflet_spy.call_args.kwargs


class TestPortrayalDictReuse:
    """A portrayal dict survives rendering and renders identically every time."""

    @staticmethod
    def _model_with_point():
        model = mesa.Model()
        model.space = mg.GeoSpace(crs="epsg:4326")
        creator = mg.AgentCreator(agent_class=mg.GeoAgent, model=model, crs="epsg:4326")
        model.space.add_agents([creator.create_agent(Point(1.0, 2.0))])
        return model

    def test_same_dict_rendered_twice_is_stable(self):
        portrayal = {"marker_type": "CircleMarker", "color": "red", "radius": 7}
        before = dict(portrayal)
        model = self._model_with_point()
        mm = MapModule(
            portrayal_method=lambda _: portrayal, tiles=xyz.OpenStreetMap.Mapnik
        )

        first = mm.render(model)["agents"][1][0]
        second = mm.render(model)["agents"][1][0]

        assert portrayal == before
        assert "marker_type" in portrayal
        assert type(first) is type(second)
        assert (first.color, first.radius) == (second.color, second.radius)

    def test_default_marker_dict_keeps_its_keys(self):
        portrayal = {"color": "red"}
        before = dict(portrayal)
        model = self._model_with_point()
        mm = MapModule(
            portrayal_method=lambda _: portrayal, tiles=xyz.OpenStreetMap.Mapnik
        )

        first = mm.render(model)["agents"][1][0]
        second = mm.render(model)["agents"][1][0]

        assert portrayal == before
        assert "marker_type" not in portrayal
        assert first.radius == second.radius == 5
