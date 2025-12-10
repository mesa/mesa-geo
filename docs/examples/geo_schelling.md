# GeoSchelling Model (Polygons)

<iframe
    width="560"
    height="315"
    src="https://www.youtube.com/embed/ZnBk_eSw0_M"
    frameborder="0"
    allowfullscreen>
</iframe>

## Summary

This is a geoversion of a simplified Schelling example. For the original implementation details please see the Mesa Schelling examples.

### GeoSpace

Instead of an abstract grid space, we represent the space using NUTS-2 regions to create the GeoSpace in the model.

### GeoAgent

NUTS-2 regions are the GeoAgents. The neighbors of a polygon are considered those polygons that touch its border (i.e., edge neighbours). During the running of the model, a polygon queries the colors of the surrounding polygon and if the ratio falls below a certain threshold (e.g., 40% of the same color), the agent moves to an uncolored polygon.

## How to Run

To run the model interactively, run `mesa runserver` in [this directory](https://github.com/mesa/mesa-examples/tree/main/gis/geo_schelling). e.g.

```bash
mesa runserver
```

Then open your browser to [http://127.0.0.1:8521/](http://127.0.0.1:8521/) and press `Start`.
