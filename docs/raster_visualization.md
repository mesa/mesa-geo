# Visualizing Raster Layers

`make_geospace_component` draws a model's {class}`~mesa_geo.geospace.GeoSpace` on a
Leaflet map. Vector layers and agents are styled through `agent_portrayal`, and raster
layers are styled through `raster_portrayal`, which colours whole bands at once with
matplotlib colormaps instead of asking for one RGBA tuple per cell.

`raster_portrayal` is keyword-only and optional. Without it, raster layers fall back to
the [legacy per-cell path](#legacy-per-cell-path), so existing models keep rendering
unchanged.

## Quick start

Pass a single {class}`~mesa.visualization.components.PropertyLayerStyle` to style every
band of every raster layer the same way:

```python
from mesa.visualization.components import PropertyLayerStyle
from mesa_geo.visualization import make_geospace_component


def agent_portrayal(agent):
    return {"color": "tab:blue"}


component = make_geospace_component(
    agent_portrayal,
    raster_portrayal=PropertyLayerStyle(colormap="viridis"),
)
```

The component is then passed to `SolaraViz` through the `components=` keyword, where
`model` and `model_params` are your own model instance and parameters:

```python
from mesa.visualization import SolaraViz

page = SolaraViz(
    model,
    name="Geo Model",
    model_params=model_params,
    components=[component],
)
```

## Styling fields

`PropertyLayerStyle` comes from mesa core, and mesa-geo honours these fields:

| Field | Meaning |
| --- | --- |
| `colormap` | A matplotlib colormap applied across the band's normalized values. |
| `color` | A single colour, ramped by opacity instead of hue. Used when `colormap` is not set. |
| `vmin` / `vmax` | Explicit normalization bounds. When omitted, the band's finite min/max are used. |
| `alpha` | Overlay opacity, default `0.8`. |

Set exactly one of `colormap` or `color`. `PropertyLayerStyle` raises `ValueError` when it
is created with both or with neither.

```{note}
`PropertyLayerStyle.colorbar` is accepted by the dataclass but ignored by the Leaflet
renderer, so no colour legend is drawn. A map legend is an open request in
[mesa-geo#97](https://github.com/mesa/mesa-geo/issues/97).
```

### Colormaps

`colormap` accepts three forms, shown below in order: a registered colormap name, a
`Colormap` instance, and a non-empty sequence of colours.

```python
from matplotlib import colormaps
from mesa.visualization.components import PropertyLayerStyle

PropertyLayerStyle(colormap="terrain")
PropertyLayerStyle(colormap=colormaps["terrain"])
PropertyLayerStyle(colormap=["#f7fbff", "#08306b"])
```

A sequence is interpolated into a `LinearSegmentedColormap`.

Leaving `colormap` unset, which is the same as `colormap=None`, is not an error: it
selects the single-colour path described below, which draws with `color` instead. A
`colormap` that is set but invalid is caught when the layer is rendered, not when the
style is created. An empty string, an empty sequence or a number raises `TypeError` naming
the layer and band, and a string that is not a registered colormap name raises
matplotlib's `KeyError`.

### Single-colour bands

With `color` instead of `colormap`, the band is drawn in one hue and the *value* drives
opacity, so high cells read as more solid:

```python
from mesa.visualization.components import PropertyLayerStyle

PropertyLayerStyle(color="tab:red", vmin=0, vmax=100)
```

The colour's own alpha, the normalized value, and `alpha` are multiplied together.

## Normalization

When `vmin` or `vmax` is omitted, the band's finite minimum and maximum are used, so the
full colormap range is spent on the data actually present. Supplying both pins the scale,
which is what you want when comparing frames across a run. Otherwise the ramp re-scales
every step and a static field appears to shift.

```{note}
`vmin=0` is honoured. mesa-geo tests these bounds with `is not None` rather than the
truthiness check mesa core uses, so a zero lower bound is kept instead of being replaced
by the band's own minimum. This is a deliberate divergence from core.
```

`vmin > vmax` raises `ValueError`. If either resolved bound is non-finite, the band is
skipped rather than drawn wrong.

## Missing data and degenerate bands

Raster bands routinely carry nodata as `NaN`, and the renderer handles it without any
masking work on your side:

- **`NaN` cells are drawn fully transparent.** Alpha is forced to zero after colouring, so
  nodata never picks up the colormap's low-end colour, and the basemap shows through.
- **A band that is entirely `NaN` is skipped.** No overlay is emitted for it at all.
- **A band whose values are all identical stays visible.** The alpha ramp is meaningless
  when `vmin == vmax`, so on the `color` path the renderer substitutes a flat ramp of `1.0`
  instead of producing an invisible layer. The result is still scaled by the colour's own
  alpha and by `alpha`, so a constant band at default settings draws at `0.8`.

## Band draw order

Bands are drawn in the order they were first added to the layer, so the last band added
sits on top. Replacing an existing band's values with `set_band` keeps its position, while
removing a band with `remove_band` and adding it again moves it to the top. The order of
`RasterLayer.attributes` does not reflect draw order, because it is a set.

To put a band on top, add it last. Here `slope` is drawn above `elevation`:

```python
import mesa
import numpy as np
from mesa_geo import RasterLayer

layer = RasterLayer(
    width=20,
    height=20,
    crs="epsg:4326",
    total_bounds=[0, 0, 20, 20],
    model=mesa.Model(),
)
layer.set_band("elevation", np.random.rand(20, 20) * 3000)
layer.set_band("slope", np.random.rand(20, 20) * 45)
```

Each render reads the layer's current band values, so updating cell attributes in
`model.step()`, or replacing a whole band there with `set_band`, shows up in the next
frame.

## Per-band styling

Pass a callable instead of a single style to vary styling by layer and band. It receives
`(layer_name, band_name)` and returns a `PropertyLayerStyle`, or `None` to skip that band.
`agent_portrayal` here is the function from the quick start:

```python
from mesa.visualization.components import PropertyLayerStyle
from mesa_geo.visualization import make_geospace_component


def raster_portrayal(layer_name, band_name):
    if band_name == "elevation":
        return PropertyLayerStyle(colormap="terrain", vmin=0, vmax=3000)
    if band_name == "slope":
        return PropertyLayerStyle(colormap="Reds")
    return None


component = make_geospace_component(
    agent_portrayal,
    raster_portrayal=raster_portrayal,
)
```

Returning something that is neither a `PropertyLayerStyle` nor `None` raises `TypeError`
identifying the layer and band.

### What `layer_name` is

`layer_name` is the name the {class}`~mesa_geo.geospace.GeoSpace` resolves for the layer.
A layer registered with `space.add_layer(layer, name="terrain")` is reported under that
name. A layer added without one falls back to the layer's own `name` attribute, so
`RasterLayer(..., name="terrain")` is still reported as `"terrain"` even though it was
never registered. `layer_name` is `None` only when the layer has neither a registered name
nor a `name` attribute.

Because `None` is possible, branch on `band_name` when you have not named your layers, or
name them so the callable can tell them apart.

(legacy-per-cell-path)=
## The legacy per-cell path

Omitting `raster_portrayal` keeps the original behaviour: `agent_portrayal` is called for
each {class}`~mesa_geo.raster_layers.Cell` and should return an `(r, g, b, a)` tuple.

```python
import mesa_geo as mg
from mesa_geo.visualization import make_geospace_component


def agent_portrayal(agent):
    if isinstance(agent, mg.Cell):
        return (255, 0, 0, 1)
    return {"radius": 4, "color": "tab:blue"}


component = make_geospace_component(agent_portrayal)
```

This path still works and is still tested. It calls your function once per cell. On a
1000 by 1000 layer with a trivial per-cell function, a full render on this path took about
1.8 times as long as on the vectorized path. Prefer `raster_portrayal` for new models.

If a model has a `RasterLayer` and `make_geospace_component()` is called with neither
`agent_portrayal` nor `raster_portrayal`, rendering the map raises `ValueError` rather than
silently drawing nothing.

`ImageLayer` is unaffected by `raster_portrayal`: it already holds RGBA values and is
drawn directly on either path.

## Agent colours

Colours in the `agent_portrayal` dictionary are resolved through matplotlib, so any
matplotlib colour specification works where Leaflet expects a CSS colour. That includes
named colours, the `tab:` and `xkcd:` families, the cycle colours `C0` through `C9`, and
hex strings:

```python
def agent_portrayal(agent):
    return {"color": "tab:blue", "fillColor": "xkcd:sky blue"}
```

The `color`, `fillColor` and `fill_color` keys are converted; `fill_color` is normalized to
`fillColor` for Leaflet. For lines and polygons the value becomes a hex string, keeping
8-digit hex when the colour carries alpha. For point markers the colour is split into an
opaque hex value plus a matching `opacity` or `fill_opacity`, because ipyleaflet's `Color`
trait rejects 8-digit hex on older ipywidgets.

Values matplotlib cannot parse are passed through untouched, so CSS-only keywords still
reach Leaflet. `"none"` and `"transparent"` are deliberately preserved rather than
normalized to a transparent hex, since Leaflet hit-tests `fill: none` differently from a
fully transparent fill.

## Current limitations

- `PropertyLayerStyle.colorbar` is ignored; no legend is drawn.
- The band read is still a Python-level loop over cells, so the pipeline is fast at the
  colouring step but not vectorized end to end.
- Reprojection and PNG encoding are the same on both paths, and together they take most
  of the render time on large layers.
