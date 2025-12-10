# GeoSchelling Model (Points & Polygons)

<iframe
    width="560"
    height="315"
    src="https://www.youtube.com/embed/iLMU6jfmir8"
    frameborder="0"
    allowfullscreen>
</iframe>

## Summary

This is a geoversion of a simplified Schelling example.

### GeoSpace

The NUTS-2 regions are considered as a shared definition of neighborhood among all people agents, instead of a locally defined neighborhood such as Moore or von Neumann.

### GeoAgent

There are two types of GeoAgents: people and regions. Each person resides in a randomly assigned region, and checks the color ratio of its region against a pre-defined "happiness" threshold at every time step. If the ratio falls below a certain threshold (e.g., 40%), the agent is found to be "unhappy", and randomly moves to another region. People are represented as points, with locations randomly chosen within their regions. The color of a region depends on the color of the majority population it contains (i.e., point in polygon calculations).

## How to run

To run the model interactively, run `mesa runserver` in [this directory](https://github.com/mesa/mesa-examples/tree/main/gis/geo_schelling_points). e.g.

```bash
mesa runserver
```

Then open your browser to [http://127.0.0.1:8521/](http://127.0.0.1:8521/) and press `Start`.
