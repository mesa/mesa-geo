import dataclasses
import warnings
from dataclasses import dataclass

import geopandas as gpd
import ipyleaflet
import numpy as np
import solara
import xyzservices
from folium.utilities import image_to_url
from matplotlib import colormaps, colors
from mesa.visualization.components import PropertyLayerStyle
from mesa.visualization.utils import update_counter
from shapely.geometry import Point, mapping

from mesa_geo.raster_layers import ImageLayer, RasterBase, RasterLayer
from mesa_geo.tile_layers import LeafletOption, RasterWebTile

# Module-level rather than per-instance: GeoSpaceLeaflet builds a fresh MapModule
# on every render, so instance state would warn once per simulation step.
_AGENT_PORTRAYAL_STATE = {"warned": False}


def make_geospace_leaflet(
    agent_portrayal,
    view=None,
    tiles=xyzservices.providers.OpenStreetMap.Mapnik,
    **kwargs,
):
    warnings.warn(
        "make_geospace_leaflet is deprecated, use make_geospace_component instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return make_geospace_component(agent_portrayal, view, tiles, **kwargs)


def make_geospace_component(
    agent_portrayal=None,
    view=None,
    tiles=xyzservices.providers.OpenStreetMap.Mapnik,
    *,
    raster_portrayal=None,
    **kwargs,
):
    """
    Create a Solara component that displays a Leaflet map for a model's GeoSpace.

    This function returns a factory callable that can be supplied to Mesa's
    `SolaraViz` to embed an interactive Leaflet map showing the model's
    :class:`~mesa_geo.geospace.GeoSpace`. The map is rendered using ipyleaflet
    and will draw raster layers, vector layers, and agents with their portrayals,
    using a user-provided `agent_portrayal` function.

    For a raster Cell, the portrayal method should return a (r, g, b, a) tuple.

    For a GeoAgent, the portrayal method should return a dictionary.
        - For a Line or a Polygon, the available options can be found at: https://leafletjs.com/reference.html#path-option
        - For a Point, the available options can be found at: https://leafletjs.com/reference.html#circlemarker-option
        - In addition, the portrayal dictionary can contain a "description" key, which will be used as the popup text.

    :param agent_portrayal: A method that takes a GeoAgent (or a Cell) and returns
        a dictionary of options (or a (r, g, b, a) tuple) for Leaflet.js.
        When ``raster_portrayal`` is provided, this function only needs to handle
        GeoAgents; raster styling is handled separately.
    :param view: Initial map center as ``(latitude, longitude)``. If not provided,
        the map is centered from ``model.space.total_bounds``.
    :param tiles: An optional tile layer to use. Can be a :class:`RasterWebTile` or
        a :class:`xyzservices.TileProvider`. Default is `xyzservices.providers.OpenStreetMap.Mapnik`.

        If the tile provider requires registration, you can pass the API key inside
        the `options` parameter of the :class:`RasterWebTile` constructor.

        For example, to use the `Mapbox` raster tile provider, you can use:

        .. code-block:: python

            import mesa_geo as mg

            mg.RasterWebTile(
                url="https://api.mapbox.com/v4/mapbox.satellite/{z}/{x}/{y}.png?access_token={access_token}",
                options={
                    "access_token": "my-private-ACCESS_TOKEN",
                    "attribution": '&copy; <a href="https://www.mapbox.com/about/maps/" target="_blank">Mapbox</a> &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors <a href="https://www.mapbox.com/map-feedback/" target="_blank">Improve this map</a>',
                },
            )

        Note that `access_token` can have different names depending on the provider,
        e.g., `api_key` or `key`. You can check the documentation of the provider
        for more details.

        `xyzservices` provides a list of providers requiring registration as well:
        https://xyzservices.readthedocs.io/en/stable/registration.html

        For example, you may use the following code to use the `Mapbox` provider:

        .. code-block:: python

            import xyzservices.providers as xyz

            xyz.MapBox(id="<insert map_ID here>", accessToken="my-private-ACCESS_TOKEN")

    :param raster_portrayal: Controls how :class:`~mesa_geo.RasterLayer` bands
        are styled. Can be:

        - A single :class:`~mesa.visualization.components.PropertyLayerStyle`:
          applied to every band of every ``RasterLayer``.
        - A callable ``(layer_name, band_name) -> PropertyLayerStyle | None``:
          return a style per band, or ``None`` to skip that band.

        When omitted (the default), the legacy ``agent_portrayal`` path is used
        for raster rendering (``to_image`` with the Cell RGBA callback).

        ``PropertyLayerStyle`` fields:

        - ``colormap``: A matplotlib colormap name, ``Colormap`` object, or list
          of colors for ``LinearSegmentedColormap.from_list``.
        - ``color``: A single matplotlib-parseable color for uniform fill with
          alpha ramp.
        - ``vmin`` / ``vmax``: Explicit normalization bounds. If omitted, the
          band's finite min/max are used.
        - ``alpha``: Overlay opacity (default ``0.8``).

        ``PropertyLayerStyle.colorbar`` is currently ignored by the Leaflet
        renderer. Colorbar support is planned in a follow-up.

        .. code-block:: python

            from mesa.visualization.components import PropertyLayerStyle
            from mesa_geo.visualization import make_geospace_component

            # Single style applied to all bands
            raster_style = PropertyLayerStyle(colormap="viridis")
            component = make_geospace_component(
                agent_portrayal,
                raster_portrayal=raster_style,
            )

            # Per-band callable
            def raster_portrayal(layer_name, band_name):
                if band_name == "elevation":
                    return PropertyLayerStyle(colormap="terrain", vmin=0, vmax=3000)
                if band_name == "slope":
                    return PropertyLayerStyle(colormap="Reds")
                return None  # skip other bands

            component = make_geospace_component(
                agent_portrayal,
                raster_portrayal=raster_portrayal,
            )

    :param **kwargs: Extra keyword arguments forwarded to :class:`ipyleaflet.Map`
        (e.g., ``zoom=``, ``scroll_wheel_zoom=``). The available options can be found
        at: https://ipyleaflet.readthedocs.io/en/latest/api_reference/index.html#ipyleaflet.leaflet.Map

    :return: A factory callable to be passed as a SolaraViz component.
    :rtype: Callable[[mesa.Model], solara.Element]

    .. warning::
        When using this component with :class:`~mesa.visualization.SolaraViz`,
        pass the list of components via the ``components=`` keyword argument
        (not as a positional argument). See the SolaraViz docs:
        https://mesa.readthedocs.io/latest/apis/visualization.html

    .. rubric:: Example
    Define a custom portrayal for agents and add a map component to SolaraViz:

    .. code-block:: python

        import mesa_geo as mg
        from mesa.visualization import SolaraViz, make_plot_component
        from mesa_geo.visualization import make_geospace_component

        def agent_portrayal(agent):
            # Return Leaflet style options or RGBA tuple
            if isinstance(agent, mg.GeoAgent):
                return {"radius": 4, "color": "blue"}
            elif isinstance(agent, mg.Cell):
                return (255, 0, 0, 1)  # Red color for raster cells

        page = SolaraViz(
            model,
            name="Geo Model",
            model_params=model_params,
            components=[
                make_geospace_component(agent_portrayal),
                make_plot_component(["happy", "unhappy"]),
            ],
        )
    """

    kwargs_to_pass = {}
    if raster_portrayal is not None:
        kwargs_to_pass["raster_portrayal"] = raster_portrayal

    def MakeSpaceMatplotlib(model):
        return GeoSpaceLeaflet(
            model,
            agent_portrayal,
            view,
            tiles,
            **kwargs_to_pass,
            **kwargs,
        )

    return MakeSpaceMatplotlib


@solara.component
def GeoSpaceLeaflet(
    model, agent_portrayal, view, tiles, *, raster_portrayal=None, **kwargs
):
    update_counter.get()
    map_drawer = MapModule(
        portrayal_method=agent_portrayal,
        tiles=tiles,
        raster_portrayal=raster_portrayal,
    )
    model_view = map_drawer.render(model)

    if view is None:
        # longlat [min_x, min_y, max_x, max_y] to latlong [min_y, min_x, max_y, max_x]
        transformed_xx, transformed_yy = model.space.transformer.transform(
            xx=[model.space.total_bounds[0], model.space.total_bounds[2]],
            yy=[model.space.total_bounds[1], model.space.total_bounds[3]],
        )
        view = [
            (transformed_yy[0] + transformed_yy[1]) / 2,
            (transformed_xx[0] + transformed_xx[1]) / 2,
        ]

    layers = (
        [ipyleaflet.TileLayer.element(url=map_drawer.tiles["url"])] if tiles else []
    )
    for layer in model_view["layers"]["rasters"]:
        layers.append(
            ipyleaflet.ImageOverlay(
                url=layer["url"],
                bounds=layer["bounds"],
            )
        )
    for layer in model_view["layers"]["vectors"]:
        layers.append(ipyleaflet.GeoJSON(element=layer))
    ipyleaflet.Map.element(
        center=view,
        layers=[
            *layers,
            ipyleaflet.GeoJSON.element(data=model_view["agents"][0]),
            *model_view["agents"][1],
        ],
        **kwargs,
    )


@dataclass
class LeafletViz:
    """A dataclass defining the portrayal of a GeoAgent in Leaflet map.

    The fields are defined to be consistent with GeoJSON options in
    Leaflet.js: https://leafletjs.com/reference.html#geojson
    """

    style: dict[str, LeafletOption] | None = None
    popupProperties: dict[str, LeafletOption] | None = None  # noqa: N815


class _RasterRenderer:
    """Internal renderer for raster layers (both vectorized PropertyLayerStyle and legacy to_image).

    Separated from MapModule so that a future GeoSpaceRenderer extraction is
    mechanical: move the class out, wire it up, done.
    """

    def __init__(self, raster_portrayal=None, legacy_portrayal=None, crs="epsg:4326"):
        self.raster_portrayal = raster_portrayal
        self.legacy_portrayal = legacy_portrayal
        self._crs = crs

    def render_layer(self, layer, layer_name=None):
        """Return the image overlays for a single raster layer."""
        if self.raster_portrayal is None:
            if isinstance(layer, RasterLayer):
                if self.legacy_portrayal is None:
                    raise ValueError(
                        "Cannot render RasterLayer: neither 'raster_portrayal' nor 'agent_portrayal' was provided."
                    )
                layer = layer.to_image(colormap=self.legacy_portrayal)
            layer_to_render = layer.to_crs(self._crs)
            return [
                {
                    "url": image_to_url(layer_to_render.values.transpose([1, 2, 0])),
                    "bounds": self._bounds(layer_to_render),
                }
            ]

        if not isinstance(layer, RasterLayer):
            layer_to_render = layer.to_crs(self._crs)
            return [
                {
                    "url": image_to_url(layer_to_render.values.transpose([1, 2, 0])),
                    "bounds": self._bounds(layer_to_render),
                }
            ]

        return self._render_bands(layer, layer_name)

    def _get_style(self, layer_name, band_name):
        if callable(self.raster_portrayal):
            style = self.raster_portrayal(layer_name, band_name)
        elif isinstance(self.raster_portrayal, PropertyLayerStyle):
            style = self.raster_portrayal
        else:
            raise TypeError(
                f"'raster_portrayal' must be a callable (layer_name, band_name) -> PropertyLayerStyle | None "
                f"or a PropertyLayerStyle instance, got {type(self.raster_portrayal).__name__}."
            )

        if style is not None and not isinstance(style, PropertyLayerStyle):
            raise TypeError(
                f"Portrayal for band {band_name!r} of layer {layer_name!r} must be a PropertyLayerStyle or None, "
                f"got {type(style).__name__}."
            )
        if style is None:
            return None, None
        if style.colormap is None:
            return style, None
        return style, self._resolve_colormap(style.colormap, layer_name, band_name)

    @staticmethod
    def _resolve_colormap(colormap, layer_name, band_name):
        """Return *colormap* as a :class:`matplotlib.colors.Colormap`.

        Accepts a registered colormap name, a ``Colormap`` instance, or a
        non-empty sequence of colors. Raises :class:`TypeError` otherwise.
        """
        if isinstance(colormap, colors.Colormap):
            return colormap
        if isinstance(colormap, str) and colormap:
            return colormaps[colormap]
        if isinstance(colormap, (list, tuple)) and len(colormap):
            return colors.LinearSegmentedColormap.from_list("custom_cmap", colormap)
        raise TypeError(
            f"Invalid 'colormap' for band {band_name!r} of layer {layer_name!r}: {colormap!r}. "
            f"Expected a registered colormap name, a matplotlib Colormap instance, "
            f"or a non-empty sequence of colors."
        )

    def _render_bands(self, layer, layer_name):
        overlays = []
        # _data is a dict, so band order is insertion order; _attributes is a
        # set and would give an arbitrary z-order.
        for band_name in layer._data:
            style, cmap = self._get_style(layer_name, band_name)
            if style is None:
                continue

            # get_band() is the live read: it reconstructs the array from
            # cells so runtime mutations in model.step() are reflected.
            # (_data is only a construction-time snapshot until PR #332 lands).
            data = layer.get_band(band_name)
            rgba = self._band_rgba(data, style, cmap)
            if rgba is None:
                continue

            layer_to_render = ImageLayer(
                values=rgba.transpose([2, 0, 1]),
                crs=layer.crs,
                total_bounds=layer.total_bounds,
            ).to_crs(self._crs)
            values = layer_to_render.values.transpose([1, 2, 0])
            overlays.append(
                {
                    "url": image_to_url((np.clip(values, 0, 1) * 255).astype(np.uint8)),
                    "bounds": self._bounds(layer_to_render),
                }
            )
        return overlays

    @staticmethod
    def _band_rgba(data, style, cmap=None):
        """Colour one band, or return None when there is nothing to draw."""
        if np.all(np.isnan(data)):
            return None

        # Deliberate divergence from core: use `is not None` rather than a truthiness
        # check (core's `if portrayal.vmin`), so that `vmin=0` is preserved and not auto-ranged.
        vmin = style.vmin if style.vmin is not None else np.nanmin(data)
        vmax = style.vmax if style.vmax is not None else np.nanmax(data)
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            return None
        if vmin > vmax:
            raise ValueError(
                f"'vmin' ({vmin}) must be less than or equal to 'vmax' ({vmax})."
            )

        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        if cmap is not None:
            rgba = cmap(norm(data))
            rgba[..., 3] *= style.alpha
        else:
            red, green, blue, alpha = colors.to_rgba(style.color)
            rgba = np.zeros((*data.shape, 4))
            rgba[..., :3] = (red, green, blue)
            # When vmin == vmax, the alpha ramp is meaningless, so use full alpha
            ramp = np.ones_like(data, dtype=float) if vmin == vmax else norm(data)
            rgba[..., 3] = alpha * ramp * style.alpha

        rgba[np.isnan(data), 3] = 0.0
        return np.clip(rgba, 0, 1)

    @staticmethod
    def _bounds(layer):
        # longlat [min_x, min_y, max_x, max_y] to latlong [[min_y, min_x], [max_y, max_x]]
        return [
            [layer.total_bounds[1], layer.total_bounds[0]],
            [layer.total_bounds[3], layer.total_bounds[2]],
        ]


class _VectorRenderer:
    """Internal renderer for vector layers (GeoDataFrame) and GeoAgents.

    Separated from MapModule so that a future GeoSpaceRenderer extraction is
    mechanical: move the class out, wire it up, done.
    """

    def __init__(self, agent_portrayal, marker_factory, crs="epsg:4326"):
        self.agent_portrayal = agent_portrayal
        self._crs = crs
        self._marker_factory = marker_factory

    @staticmethod
    def _css_color(value):
        """Convert a matplotlib color specification to a hex string for Leaflet.

        If value is None or cannot be parsed by matplotlib (e.g. CSS-only keywords
        like 'transparent'), return the original value.
        Preserves alpha (8-digit hex) when transparency is encoded.

        ``"none"`` and ``"transparent"`` are passed through: Leaflet treats
        ``fill: none`` and a fully transparent fill differently for hit-testing,
        so they are not interchangeable with ``#00000000``. This applies to the
        GeoJSON path only; ipyleaflet's ``Color`` trait rejects the keywords, so
        markers still resolve them to a transparent colour.
        """
        if value is None:
            return None
        if isinstance(value, str) and value.strip().lower() in ("none", "transparent"):
            return value
        try:
            rgba = colors.to_rgba(value)
            has_alpha = rgba[3] < 1.0 or (
                isinstance(value, str) and len(value) == 9 and value.startswith("#")
            )
            return colors.to_hex(rgba, keep_alpha=has_alpha)
        except (ValueError, TypeError):
            return value

    @staticmethod
    def _split_alpha(value):
        """Split a color specification into an opaque hex color and its alpha.

        Returns ``(color, alpha)``, where *alpha* is ``None`` when the color is
        fully opaque or cannot be parsed by matplotlib.
        """
        if value is None:
            return None, None
        try:
            rgba = colors.to_rgba(value)
        except (ValueError, TypeError):
            return value, None
        alpha = float(rgba[3])
        return colors.to_hex(rgba, keep_alpha=False), None if alpha >= 1.0 else alpha

    def render_layer(self, layer):
        """Render a GeoDataFrame layer to geo_interface."""
        return layer.to_crs(self._crs)[["geometry"]].__geo_interface__

    def render_agents(self, agents, transformer):
        if (
            self.agent_portrayal is None
            and len(agents)
            and not _AGENT_PORTRAYAL_STATE["warned"]
        ):
            warnings.warn(
                f"The space has {len(agents)} agent(s) but no 'agent_portrayal' was "
                f"provided, so no agents will be drawn. Pass 'agent_portrayal' to draw them.",
                UserWarning,
                stacklevel=2,
            )
            _AGENT_PORTRAYAL_STATE["warned"] = True

        feature_collection = {"type": "FeatureCollection", "features": []}
        point_markers = []
        agent_portrayal = {}
        for agent in agents:
            transformed_geometry = agent.get_transformed_geometry(transformer)

            if self.agent_portrayal:
                properties = dict(self.agent_portrayal(agent))
                agent_portrayal = LeafletViz(
                    popupProperties=properties.pop("description", None)
                )

                if isinstance(agent.geometry, Point):
                    location = mapping(transformed_geometry)
                    # for some reason points are reversed
                    location = (location["coordinates"][1], location["coordinates"][0])
                    point_markers.append(self._marker_factory(location, properties))
                else:
                    for key in ("color", "fillColor", "fill_color"):
                        if key in properties:
                            properties[key] = self._css_color(properties[key])

                    if "fill_color" in properties and "fillColor" not in properties:
                        properties["fillColor"] = properties.pop("fill_color")
                    agent_portrayal.style = properties
                    agent_portrayal = dataclasses.asdict(
                        agent_portrayal,
                        dict_factory=lambda x: {k: v for (k, v) in x if v is not None},
                    )

                    feature_collection["features"].append(
                        {
                            "type": "Feature",
                            "geometry": mapping(transformed_geometry),
                            "properties": agent_portrayal,
                        }
                    )
        return [feature_collection, point_markers]


class MapModule:
    """A MapModule for Leaflet maps that uses a user-defined portrayal method
    to generate a portrayal of a raster Cell or a GeoAgent.

    For a raster Cell, the portrayal method should return a (r, g, b, a) tuple.

    For a GeoAgent, the portrayal method should return a dictionary.
        - For a Line or a Polygon, the available options can be found at: https://leafletjs.com/reference.html#path-option
        - For a Point, the available options can be found at: https://leafletjs.com/reference.html#circlemarker-option
        - In addition, the portrayal dictionary can contain a "description" key, which will be used as the popup text.
    """

    def __init__(
        self,
        portrayal_method,
        tiles,
        *,
        raster_portrayal=None,
    ):
        """Create a new MapModule.

        :param portrayal_method: A method that takes a GeoAgent (or a Cell) and returns
            a dictionary of options (or a (r, g, b, a) tuple) for Leaflet.js.
        :param tiles: An optional tile layer to use. Can be a :class:`RasterWebTile` or
            a :class:`xyzservices.TileProvider`. Default is `xyzservices.providers.OpenStreetMap.Mapnik`.

            If the tile provider requires registration, you can pass the API key inside
            the `options` parameter of the :class:`RasterWebTile` constructor.

            For example, to use the `Mapbox` raster tile provider, you can use:

            .. code-block:: python

                import mesa_geo as mg

                mg.RasterWebTile(
                    url="https://api.mapbox.com/v4/mapbox.satellite/{z}/{x}/{y}.png?access_token={access_token}",
                    options={
                        "access_token": "my-private-ACCESS_TOKEN",
                        "attribution": '&copy; <a href="https://www.mapbox.com/about/maps/" target="_blank">Mapbox</a> &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors <a href="https://www.mapbox.com/map-feedback/" target="_blank">Improve this map</a>',
                    },
                )

            Note that `access_token` can have different names depending on the provider,
            e.g., `api_key` or `key`. You can check the documentation of the provider
            for more details.

            `xyzservices` provides a list of providers requiring registration as well:
            https://xyzservices.readthedocs.io/en/stable/registration.html

            For example, you may use the following code to use the `Mapbox` provider:

            .. code-block:: python

                import xyzservices.providers as xyz

                xyz.MapBox(id="<insert map_ID here>", accessToken="my-private-ACCESS_TOKEN")

        :param raster_portrayal: A :class:`~mesa.visualization.components.PropertyLayerStyle`
            or a callable ``(layer_name, band_name) -> PropertyLayerStyle | None``
            controlling how :class:`~mesa_geo.RasterLayer` bands are colormapped.
            When omitted, the legacy ``portrayal_method`` Cell RGBA path is used.
            See :func:`make_geospace_component` for full details and examples.
        """
        self.portrayal_method = portrayal_method
        self.raster_portrayal = raster_portrayal
        self._crs = "epsg:4326"

        self.raster_renderer = _RasterRenderer(
            raster_portrayal=self.raster_portrayal,
            legacy_portrayal=self.portrayal_method,
            crs=self._crs,
        )
        self.vector_renderer = _VectorRenderer(
            agent_portrayal=self.portrayal_method,
            crs=self._crs,
            marker_factory=self._get_marker,
        )

        if isinstance(tiles, xyzservices.TileProvider):
            tiles = RasterWebTile.from_xyzservices(tiles).to_dict()
        self.tiles = tiles

    def render(self, model):
        return {
            "layers": self._render_layers(model),
            "agents": self._render_agents(model),
        }

    def _render_layers(self, model):
        layers = {"rasters": [], "vectors": [], "total_bounds": []}
        name_lookup = getattr(model.space, "_name_for_layer", lambda _: None)
        for layer in model.space.layers:
            if isinstance(layer, RasterBase):
                layer_name = name_lookup(layer)
                layers["rasters"].extend(
                    self.raster_renderer.render_layer(layer, layer_name)
                )
            elif isinstance(layer, gpd.GeoDataFrame):
                layers["vectors"].append(self.vector_renderer.render_layer(layer))
        # longlat [min_x, min_y, max_x, max_y] to latlong [min_y, min_x, max_y, max_x]
        if model.space.total_bounds is not None:
            transformed_xx, transformed_yy = model.space.transformer.transform(
                xx=[model.space.total_bounds[0], model.space.total_bounds[2]],
                yy=[model.space.total_bounds[1], model.space.total_bounds[3]],
            )
            layers["total_bounds"] = [
                [transformed_yy[0], transformed_xx[0]],  # min_y, min_x
                [transformed_yy[1], transformed_xx[1]],  # max_y, max_x
            ]
        return layers

    def _get_marker(self, location, properties):
        """
        takes point objects and transforms them to ipyleaflet marker objects

        allowed marker types are point marker types from ipyleaflet
        https://ipyleaflet.readthedocs.io/en/latest/layers/index.html

        default is circle with radius 5

        Parameters
        ----------
        location: iterable
            iterable of location in models geometry

        properties : dict
            properties passed in through agent portrayal


        Returns
        -------
        ipyleaflet marker element

        """
        marker = properties.get("marker_type", "Circle")
        options = {
            key: value for key, value in properties.items() if key != "marker_type"
        }

        if "fillColor" in options and "fill_color" not in options:
            options["fill_color"] = options.pop("fillColor")

        # ipyleaflet colours go through a traitlets Color trait, which rejected
        # 8-digit hex before ipywidgets 8.1; carry alpha as an opacity instead.
        for key, opacity_key in (("color", "opacity"), ("fill_color", "fill_opacity")):
            if key in options:
                color, alpha = _VectorRenderer._split_alpha(options[key])
                options[key] = color
                if alpha is not None and opacity_key not in options:
                    options[opacity_key] = alpha

        if "marker_type" not in properties and "radius" not in options:
            options["radius"] = 5

        if marker == "Circle":
            return ipyleaflet.Circle(location=location, **options)
        elif marker == "CircleMarker":
            return ipyleaflet.CircleMarker(location=location, **options)
        elif marker == "Marker":
            return ipyleaflet.Marker(location=location, **options)
        elif marker == "Icon":
            icon_url = options["icon_url"]
            icon_size = options.get("icon_size", [20, 20])
            icon_properties = options.get("icon_properties", {})
            icon = ipyleaflet.Icon(
                icon_url=icon_url, icon_size=icon_size, **icon_properties
            )
            return ipyleaflet.Marker(location=location, icon=icon, **options)
        elif marker == "AwesomeIcon":
            name = options["name"]
            icon_properties = options.get("icon_properties", {})
            icon = ipyleaflet.AwesomeIcon(name=name, **icon_properties)
            return ipyleaflet.Marker(location=location, icon=icon, **options)

        else:
            raise ValueError(
                f"Unsupported marker type:{marker}",
            )

    def _render_agents(self, model):
        return self.vector_renderer.render_agents(
            model.space.agents, model.space.transformer
        )
