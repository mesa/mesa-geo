# Population Model

<iframe
    width="560"
    height="315"
    src="https://www.youtube.com/embed/0k8tsYPVwQs"
    frameborder="0"
    allowfullscreen>
</iframe>

## Summary

This is an implementation of the [Uganda Example](https://github.com/abmgis/abmgis/tree/master/Chapter05-GIS/Models/UgandaExample) in Python, using [Mesa](https://github.com/mesa/mesa) and [Mesa-Geo](https://github.com/mesa/mesa-geo).

### GeoSpace

The GeoSpace consists of both a raster and a vector layer. The raster layer contains population data for each cell, and it is this data that is used for model initialisation, in the sense creating the agents. The vector layer shown in blue color represents a lake in Uganda. It overlays with the raster layer to mask out the cells that agents cannot move into.

### GeoAgent

The GeoAgents are people, created based on the population data. As this is a simple example model, the agents only move randomly to neighboring cells at each time step. To make the simulation more realistic and visually appealing, the agents in the same cell have a randomized position within the cell, so that they don’t stand on top of each other at exactly the same coordinate.

## How to run

To run the model interactively, run `mesa runserver` in [this directory](https://github.com/mesa/mesa-examples/tree/main/gis/population). e.g.

```bash
mesa runserver
```

Then open your browser to [http://127.0.0.1:8521/](http://127.0.0.1:8521/) and press `Start`.
