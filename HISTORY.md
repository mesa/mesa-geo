Release History
---------------

# 0.9.2 (2026-01-05)

## What's Changed
### 📜 Documentation improvements
* Replace CITATION.bib with CITATION.cff for better GitHub integration by @EwoutH in https://github.com/mesa/mesa-geo/pull/272
* Update organization name from `projectmesa` to `mesa` by @EwoutH in https://github.com/mesa/mesa-geo/pull/289
* Update contact email. by @jackiekazil in https://github.com/mesa/mesa-geo/pull/290
* docs: replace black with ruff in contribution guide by @falloficarus22 in https://github.com/mesa/mesa-geo/pull/292
### 🔧 Maintenance
* pin mesa to <3.3 due to solaraviz compatibility by @wang-boyu in https://github.com/mesa/mesa-geo/pull/294

## New Contributors
* @jackiekazil made their first contribution in https://github.com/mesa/mesa-geo/pull/290
* @falloficarus22 made their first contribution in https://github.com/mesa/mesa-geo/pull/292

**Full Changelog**: https://github.com/mesa/mesa-geo/compare/v0.9.1...v0.9.2

# 0.9.1 (2025-02-04)

## What's Changed

### 🛠 Enhancements made
* refactor: :recycle: separate cells' initialization into a private method by @SongshGeo in https://github.com/mesa/mesa-geo/pull/274
### 🐛 Bugs fixed
* fix typo in intro tutorial by @wang-boyu in https://github.com/mesa/mesa-geo/pull/275

## New Contributors
* @SongshGeo made their first contribution in https://github.com/mesa/mesa-geo/pull/274

**Full Changelog**: https://github.com/mesa/mesa-geo/compare/v0.9.0...v0.9.1

# 0.9.0 (2024-12-21)

## What's Changed

### 🐛 Bugs fixed

* fix links to readthedocs site by @wang-boyu in https://github.com/mesa/mesa-geo/pull/257
* fix broken mesa dependencies in GeoJupyterViz by @AdamZh0u in https://github.com/mesa/mesa-geo/pull/269

### 🔧 Maintenance

* rename make_geospace_leaflet to make_geospace_component by @wang-boyu in https://github.com/mesa/mesa-geo/pull/270
* update make_plot_measure method name from mesa viz by @wang-boyu in https://github.com/mesa/mesa-geo/pull/264
* Require Mesa 3.0 stable by @EwoutH in https://github.com/mesa/mesa-geo/pull/260

**Full Changelog**: https://github.com/mesa/mesa-geo/compare/v0.9.0a1...v0.9.0

# 0.9.0a1 (2024-10-17)

## Highlights
This small pre-release fixes a bug in the RasterLayer rendering and deprecated the old GeoJupyterViz, in favor of the new SolaraViz.

## What's Changed

### 🐛 Bugs fixed
* fix raster layer rendering in solaraviz by @wang-boyu in https://github.com/mesa/mesa-geo/pull/254
### 📜 Documentation improvements
* Deprecate geojupyterviz and update intro tutorial by @wang-boyu in https://github.com/mesa/mesa-geo/pull/255

**Full Changelog**: https://github.com/mesa/mesa-geo/compare/v0.9.0a0...v0.9.0a1

# 0.9.0a0 (2024-09-27)
## Highlights
The Mesa-geo `v0.9.0a0` pre-release is the first Mesa-geo version compatible with Mesa 3.0.

One of the most notable changes is the automatic assignment of unique IDs to agents. This eliminates the need for manual ID specification, simplifying agent creation. For example, where you previously might have initialized an agent with:

```python
agent = MyGeoAgent(unique_id=1, model=model, geometry=point, crs="EPSG:4326")
```

You now simply omit the `unique_id`:

```python
agent = MyGeoAgent(model=model, geometry=point, crs="EPSG:4326")
```

Mesa-geo can now directly use Mesa 3.0's SolaraViz visualisation, with an additional `make_geospace_leaflet` method to support geospaces. The new visualization can be used like:

```python
from mesa.visualization import SolaraViz
import mesa_geo.visualization as mgv

model = GeoSIR()
SolaraViz(
    model,
    name="GeoSIR",
    components=[
        mgv.make_geospace_leaflet(SIR_draw, zoom=12, scroll_wheel_zoom=False),
        mesa.visualization.make_plot_measure(["infected", "susceptible", "recovered", "dead"]),
        mesa.visualization.make_plot_measure(["safe", "hotspot"]),
    ]
)
```

The `v0.9.0a0` pre-release is a snapshot release to allow starting testing against Mesa 3.0, and might introduce new breaking changes in upcoming (pre-)releases.

## What's Changed
### ⚠️ Breaking changes
* Require Mesa 3.0 by @EwoutH in https://github.com/mesa/mesa-geo/pull/244
* Automatically assign unique_id's by @EwoutH in https://github.com/mesa/mesa-geo/pull/248
### 🛠 Enhancements made
* add method to make geospace as a solara component by @wang-boyu in https://github.com/mesa/mesa-geo/pull/246
### 🐛 Bugs fixed
* raster_layer: Don't pass unique_id to Agent in Cell by @EwoutH in https://github.com/mesa/mesa-geo/pull/249
### 📜 Documentation improvements
* Readthedocs: Don't let notebook failures pass silently by @EwoutH in https://github.com/mesa/mesa-geo/pull/250
* intro tutorial: Remove unique_id from Agent init by @EwoutH in https://github.com/mesa/mesa-geo/pull/251

**Full Changelog**: https://github.com/mesa/mesa-geo/compare/v0.8.1...v0.9.0a0

# 0.8.1 (2024-09-03)
## Highlights
Mesa-Geo 0.8.1 is a small patch release containing a single feature, a documentation update and a bug fixed.

The real novelty is that from now on, all GIS examples on [Mesa-examples](https://github.com/mesa/mesa-examples) are tested in CI against Mesa-Geo. We fixed 16 bugs in the 7 GIS example models ([mesa-examples#172](https://github.com/mesa/mesa-examples/issues/172)), which are now available on two branches:
- On the `main` branch [GIS examples](https://github.com/mesa/mesa-examples/tree/main/gis) can be found will keep being updated for the latest Mesa and Mesa-Geo versions.
- On the `mesa-2.x` branch [GIS examples](https://github.com/mesa/mesa-examples/tree/mesa-2.x/gis) examples can be found that keep working with Mesa 2.x and Mesa-Geo 0.8.x.

The Mesa-Geo 0.8.x. series is compatible with Mesa 2.3.x. The next Mesa-Geo release series, 0.9.x, will be compatible with with Mesa 3.0.

## What's Changed
### 🎉 New features added
* Expose rasterio's opener argument in Rasterlayer.from_file by @EwoutH in https://github.com/mesa/mesa-geo/pull/237
### 🐛 Bugs fixed
* add model parameter in RasterLayer class method by @wang-boyu in https://github.com/mesa/mesa-geo/pull/240
### 📜 Documentation improvements
* Update intro_tutorial.ipynb by @tpike3 in https://github.com/mesa/mesa-geo/pull/234
### 🔧 Maintenance
* Add test script for GIS examples and run that in CI by @EwoutH in https://github.com/mesa/mesa-geo/pull/241

**Full Changelog**: https://github.com/mesa/mesa-geo/compare/v0.8.0...v0.8.1

# 0.8.0 (2024-08-21)

## Highlights

- The Tornado visualization server is removed and replaced with SolaraViz, which also works within Jupyter notebooks (https://github.com/mesa/mesa-geo/pull/212). This is in line with Mesa's recent changes to use Solara for visualization.
- The [Introductory Tutorial](https://mesa-geo.readthedocs.io/en/stable/tutorials/intro_tutorial.html) has been fully rewritten for Mesa-Geo 0.8.0
- The 0.8.x series are the releases compatible with Mesa 2.3.x. The next major release will be compatible with Mesa 3.0+.

### 🎉 New features added

* Update mesa-geo to sync with mesa >=2.3.0 by @tpike3 in https://github.com/mesa/mesa-geo/pull/212

### 🛠 Enhancements made

* Update tutorial and viz by @tpike3 in https://github.com/mesa/mesa-geo/pull/217

### 📜 Documentation improvements

* fix links and installation instructions in README file by @wang-boyu in https://github.com/mesa/mesa-geo/pull/213
* .readthedocs.yaml: Use latest Ubuntu and Python versions by @EwoutH in https://github.com/mesa/mesa-geo/pull/221
* docs: update conf.py to be in sync with mesa by @wang-boyu in https://github.com/mesa/mesa-geo/pull/223
* docs: remove api docs entry for removed visualization module by @wang-boyu in https://github.com/mesa/mesa-geo/pull/224
* Fix kernel issue by @tpike3 in https://github.com/mesa/mesa-geo/pull/229
* Remove cell output by @tpike3 in https://github.com/mesa/mesa-geo/pull/231

### 🔧 Maintenance

* Update configuration, metadata and tests by @tpike3 in https://github.com/mesa/mesa-geo/pull/208
* fix: Use correct package name for Pip by @rht in https://github.com/mesa/mesa-geo/pull/214
* pyproject.toml: Always use latest ruff by @EwoutH in https://github.com/mesa/mesa-geo/pull/219
* pyproject.toml: Use mesa version smaller than 3 for now by @EwoutH in https://github.com/mesa/mesa-geo/pull/220
* CI: Add job to test with pre-release dependencies, including Mesa by @EwoutH in https://github.com/mesa/mesa-geo/pull/218

**Full Changelog**: https://github.com/mesa/mesa-geo/compare/v0.7.1...v0.8.0

## 0.7.1 (2024-03-27)

### 🐛 Bugs fixed

* fix: remove old map layers before rendering new layers by @wang-boyu in https://github.com/mesa/mesa-geo/pull/194 (thanks @rw73mg for reporting)

**Full Changelog**: https://github.com/mesa/mesa-geo/compare/v0.7.0...v0.7.1

## 0.7.0 (2024-01-17)

### Special Notes

- Update Mesa dependency to v2.2
- The pinning of Mesa is now on the major version, instead of the minor version. This means that Mesa-Geo v0.7.0 will work with Mesa v2.2, v2.3, v2.4, etc. but not with Mesa v3.0 or later.

### 🛠 Enhancements made

* create and update rtree spatial index only when needed by @wang-boyu in https://github.com/mesa/mesa-geo/pull/179

### 🔧 Maintenance

* fix link to examples by @wang-boyu in https://github.com/mesa/mesa-geo/pull/167
* Correct link to GeoSchelling example and update copyright string by @Holzhauer in https://github.com/mesa/mesa-geo/pull/175
* fix rtd build error and upgrade to python 3.9 by @wang-boyu in https://github.com/mesa/mesa-geo/pull/176
* update pre-commit and ga workflows to be consistent with mesa by @wang-boyu in https://github.com/mesa/mesa-geo/pull/181
* add config file to automatically generate release notes by @wang-boyu in https://github.com/mesa/mesa-geo/pull/184
* update ga workflows to be consistent with mesa by @wang-boyu in https://github.com/mesa/mesa-geo/pull/185

## New Contributors

* @Holzhauer made their first contribution in https://github.com/mesa/mesa-geo/pull/175

**Full Changelog**: https://github.com/mesa/mesa-geo/compare/v0.6.0...v0.7.0

## 0.6.0 (2023-09-13)

### Special Notes

- update mesa dependency to v2.1

### Improvements

- use Pathlib [#149](https://github.com/mesa/mesa-geo/pull/149) (thanks @catherinedevlin for contributing)

- ***Docs updates***
  - docs: use pydata theme [#152](https://github.com/mesa/mesa-geo/pull/152)
  - docs: use myst-nb to compile notebooks at build time [#159](https://github.com/mesa/mesa-geo/pull/159)

- ***Example updates***
  - remove examples and their tests [#163](https://github.com/mesa/mesa-geo/pull/163)

### Fixes

- fix AttributeError in GeoSpace.agents_at() [#165](https://github.com/mesa/mesa-geo/pull/165) (thanks @SongshGeo for reporting)

## 0.5.0 (2023-03-09)

### Improvements

- ***Docs updates***
  - add citation information about mesa-geo [#117](https://github.com/mesa/mesa-geo/pull/117)
  - add citation info to readthedocs [#118](https://github.com/mesa/mesa-geo/pull/118)
  - docs: update docstrings on how to use providers requiring registration [#141](https://github.com/mesa/mesa-geo/pull/141)

- ***Front-end updates***
  - add scale to Leaflet map [#123](https://github.com/mesa/mesa-geo/pull/123)
  - allow basemap tiles configuration [#127](https://github.com/mesa/mesa-geo/pull/127)

- ***CI updates***
  - add testing for python 3.11 [#122](https://github.com/mesa/mesa-geo/pull/122)
  - ci: replace flake8 with ruff [#132](https://github.com/mesa/mesa-geo/pull/132)
  - ci: update os, python versions, and dependabot configurations [#142](https://github.com/mesa/mesa-geo/pull/142)
  - ci: pin ruff version to v0.0.254 [#144](https://github.com/mesa/mesa-geo/pull/144)

### Fixes

- fix WMSWebTile.to_dict() method [#140](https://github.com/mesa/mesa-geo/pull/140)

## 0.4.0 (2022-10-18)

### Improvements

- export geoagents and raster cells [#98](https://github.com/mesa/mesa-geo/pull/98)
- use ModularServer from Mesa [#109](https://github.com/mesa/mesa-geo/pull/109)
- implement simpler Mesa-Geo namespace [#115](https://github.com/mesa/mesa-geo/pull/115)

- ***Docs updates***
  - create Read the Docs [#99](https://github.com/mesa/mesa-geo/pull/99)
  - update README with badges and matrix chat link [#100](https://github.com/mesa/mesa-geo/pull/100)

- ***Front-end updates***
  - auto zoom to geospace when view & zoom are missing [#103](https://github.com/mesa/mesa-geo/pull/103)

- ***CI updates***
  - add pre-commit config and run it on all files [#107](https://github.com/mesa/mesa-geo/pull/107)

- ***Example updates***
  - link example models to readthedocs [#101](https://github.com/mesa/mesa-geo/pull/101)
  - fix spatial variation of water level in rainfall example [#108](https://github.com/mesa/mesa-geo/pull/108)
  - fix youtube links in geo_schelling examples [#113](https://github.com/mesa/mesa-geo/pull/113)

### Fixes

- replace BuildCommand & DevelopCommand with BuildPyCommand during setup [#106](https://github.com/mesa/mesa-geo/pull/106)

## 0.3.0 (2022-07-27)

### Special Notes

- BREAKING: rename model.grid to model.space [#40](https://github.com/mesa/mesa-geo/pull/40)
- BREAKING: rename GeoAgent's shape attribute to geometry [#57](https://github.com/mesa/mesa-geo/pull/57)

### Improvements

- feat/crs [#58](https://github.com/mesa/mesa-geo/pull/58)
  - add GeoAgent.crs attribute
  - update GeoSpace with GeoAgent.crs
- extract an _AgentLayer from GeoSpace [#62](https://github.com/mesa/mesa-geo/pull/62)
- add layers into geospace [#67](https://github.com/mesa/mesa-geo/pull/67)
- implement RasterLayer [#75](https://github.com/mesa/mesa-geo/pull/75)
- create raster layer from file [#92](https://github.com/mesa/mesa-geo/pull/92)

- ***Front-end updates***
  - implement LeafletPortrayal dataclass for GeoAgent portrayal [#84](https://github.com/mesa/mesa-geo/pull/84)

- ***CI updates***
  - ci: Replace Travis with GH Actions [#47](https://github.com/mesa/mesa-geo/pull/47)
  - ci: Disable PyPy tests for now [#56](https://github.com/mesa/mesa-geo/pull/56)

- ***Dependency updates***
  - Frontend dependencies [#54](https://github.com/mesa/mesa-geo/pull/54)
    - remove all frontend dependencies available from mesa
    - create setup.cfg and pyproject.toml from setup.py
  - download leaflet during install [#59](https://github.com/mesa/mesa-geo/pull/59)
  - remove version number from leaflet filenames [#61](https://github.com/mesa/mesa-geo/pull/61)
  - update for Mesa v1.0.0 [#78](https://github.com/mesa/mesa-geo/pull/78)
    - specify mesa 1.x dependency
    - update for mesa css includes
    - remove jQuery usage in MapModule.js
    - use Slider instead of UserSettableParameter in examples

- ***Example updates***
  - update examples [#74](https://github.com/mesa/mesa-geo/pull/74)
    - change examples folder structure
    - add test for examples
    - add geo_schelling_points example
  - add rainfall and urban growth examples [#80](https://github.com/mesa/mesa-geo/pull/80)
  - add uganda example [#90](https://github.com/mesa/mesa-geo/pull/90)

- ***Other improvements***
  - add github issue templates [#38](https://github.com/mesa/mesa-geo/pull/38)
  - apply Black to all Python files [#50](https://github.com/mesa/mesa-geo/pull/50)
  - add code of conduct and contributing guide [#69](https://github.com/mesa/mesa-geo/pull/69)
  - update license with year and contributors [#86](https://github.com/mesa/mesa-geo/pull/86)
  - rename master branch to main [#89](https://github.com/mesa/mesa-geo/pull/89)

### Fixes

- fix remove_agent in GeoSpace [#34](https://github.com/mesa/mesa-geo/pull/34)
- remove deprecated skip_equivalent from pyproj [#43](https://github.com/mesa/mesa-geo/pull/43)
- flake8: Fix errors [#51](https://github.com/mesa/mesa-geo/pull/51)
- rename InstallCommand to BuildCommand [#55](https://github.com/mesa/mesa-geo/pull/55)
- fix codecov and README.md [#71](https://github.com/mesa/mesa-geo/pull/71)
- use shape.centroid instead of shape.center() [#73](https://github.com/mesa/mesa-geo/pull/73)
- fix unique id exception for raster cells [#83](https://github.com/mesa/mesa-geo/pull/83)
- fix total_bounds check in GeoSpace [#88](https://github.com/mesa/mesa-geo/pull/88)
