"""
GeoSpace
--------
"""

from __future__ import annotations

import warnings

import geopandas as gpd
import numpy as np
import pyproj
from libpysal import weights
from rtree import index
from shapely.geometry import Point
from shapely.prepared import prep

from mesa_geo.geo_base import GeoBase
from mesa_geo.geoagent import GeoAgent
from mesa_geo.raster_layers import ImageLayer, RasterLayer


class GeoSpace(GeoBase):
    """
    Space used to add a geospatial component to a model.
    """

    def __init__(self, crs="epsg:3857", *, warn_crs_conversion=True):
        """
        Create a GeoSpace for GIS enabled mesa modeling.

        :param crs: The coordinate reference system of the GeoSpace.
            If `crs` is not set, epsg:3857 (Web Mercator) is used as default.
            However, this system is only accurate at the equator and errors
            increase with latitude.
        :param warn_crs_conversion: Whether to warn when converting layers and
            GeoAgents of different crs into the crs of GeoSpace. Default to
            True.
        """
        super().__init__(crs)
        self._transformer = pyproj.Transformer.from_crs(
            crs_from=self.crs, crs_to="epsg:4326", always_xy=True
        )
        self.warn_crs_conversion = warn_crs_conversion
        self._agent_layer = _AgentLayer()
        self._static_layers = []
        self._total_bounds = None  # [min_x, min_y, max_x, max_y]

    def to_crs(self, crs, inplace=False) -> GeoSpace | None:
        super()._to_crs_check(crs)
        target_crs = pyproj.CRS.from_user_input(crs)
        if inplace:
            if self.crs.is_exact_same(target_crs):
                return None
            for agent in self.agents:
                agent.to_crs(target_crs, inplace=True)
            for layer in self.layers:
                layer.to_crs(target_crs, inplace=True)

            self.crs = target_crs
            self._transformer = pyproj.Transformer.from_crs(
                crs_from=self.crs, crs_to="epsg:4326", always_xy=True
            )
            self._total_bounds = None
            self._agent_layer._invalidate()
            return None
        else:
            geospace = GeoSpace(
                crs=target_crs, warn_crs_conversion=self.warn_crs_conversion
            )
            for agent in self.agents:
                geospace.add_agents(agent.to_crs(target_crs, inplace=False))
            for layer in self.layers:
                geospace.add_layer(layer.to_crs(target_crs, inplace=False))
            return geospace

    @property
    def transformer(self):
        """
        Return the pyproj.Transformer that transforms the GeoSpace into
        epsg:4326. Mainly used for GeoJSON serialization.
        """
        return self._transformer

    @property
    def agents(self):
        """
        Return a list of all agents in the Geospace.
        """
        return self._agent_layer.agents

    @property
    def layers(self) -> list[ImageLayer | RasterLayer | gpd.GeoDataFrame]:
        """
        Return a list of all layers in the Geospace.
        """
        return self._static_layers

    @property
    def total_bounds(self) -> np.ndarray | None:
        """
        Return the bounds of the GeoSpace in [min_x, min_y, max_x, max_y] format.
        """
        if self._total_bounds is None:
            if len(self.agents) > 0:
                self._update_bounds(self._agent_layer.total_bounds)
            if len(self.layers) > 0:
                for layer in self.layers:
                    self._update_bounds(layer.total_bounds)
        return self._total_bounds

    def _update_bounds(self, new_bounds: np.ndarray) -> None:
        if new_bounds is not None:
            if self._total_bounds is not None:
                new_min_x = min(self.total_bounds[0], new_bounds[0])
                new_min_y = min(self.total_bounds[1], new_bounds[1])
                new_max_x = max(self.total_bounds[2], new_bounds[2])
                new_max_y = max(self.total_bounds[3], new_bounds[3])
                self._total_bounds = np.array(
                    [new_min_x, new_min_y, new_max_x, new_max_y]
                )
            else:
                self._total_bounds = new_bounds

    @property
    def __geo_interface__(self):
        """
        Return a GeoJSON FeatureCollection.
        """
        features = [a.__geo_interface__() for a in self.agents]
        return {"type": "FeatureCollection", "features": features}

    def add_layer(self, layer: ImageLayer | RasterLayer | gpd.GeoDataFrame) -> None:
        """Add a layer to the Geospace.

        :param ImageLayer | RasterLayer | gpd.GeoDataFrame layer: The layer to add.
        """
        if not self.crs.is_exact_same(layer.crs):
            if self.warn_crs_conversion:
                warnings.warn(
                    f"Converting {layer.__class__.__name__} from crs {layer.crs.to_string()} "
                    f"to the crs of {self.__class__.__name__} - {self.crs.to_string()}. "
                    "Please check your crs settings if this is unintended, or set `GeoSpace.warn_crs_conversion` "
                    "to `False` to suppress this warning message.",
                    UserWarning,
                    stacklevel=2,
                )
            layer.to_crs(self.crs, inplace=True)
        self._total_bounds = None
        self._static_layers.append(layer)

    def _check_agent(self, agent):
        if hasattr(agent, "geometry"):
            if not self.crs.is_exact_same(agent.crs):
                if self.warn_crs_conversion:
                    warnings.warn(
                        f"Converting {agent.__class__.__name__} from crs {agent.crs.to_string()} "
                        f"to the crs of {self.__class__.__name__} - {self.crs.to_string()}. "
                        "Please check your crs settings if this is unintended, or set `GeoSpace.warn_crs_conversion` "
                        "to `False` to suppress this warning message.",
                        UserWarning,
                        stacklevel=2,
                    )
                agent.to_crs(self.crs, inplace=True)
        else:
            raise AttributeError("GeoAgents must have a geometry attribute")

    def add_agents(self, agents):
        """Add a list of GeoAgents to the Geospace.

        GeoAgents must have a geometry attribute. This function may also be called
        with a single GeoAgent.

        :param agents: A list of GeoAgents or a single GeoAgent to be added into GeoSpace.
        :raises AttributeError: If the GeoAgents do not have a geometry attribute.
        """
        if isinstance(agents, GeoAgent):
            agent = agents
            self._check_agent(agent)
        else:
            for agent in agents:
                self._check_agent(agent)
        self._agent_layer.add_agents(agents)
        self._total_bounds = None

    def _recreate_rtree(self, new_agents=None):
        """Create a new rtree index from agents geometries."""
        self._agent_layer._recreate_rtree(new_agents)

    def remove_agent(self, agent):
        """Remove an agent from the GeoSpace."""
        self._agent_layer.remove_agent(agent)
        self._total_bounds = None

    def get_relation(self, agent, relation):
        """Return a list of related agents.

        :param GeoAgent agent: The agent to find related agents for.
        :param str relation: The relation to find. Must be one of 'intersects',
            'within', 'contains', 'touches'.
        """
        yield from self._agent_layer.get_relation(agent, relation)

    def get_intersecting_agents(self, agent):
        return self._agent_layer.get_intersecting_agents(agent)

    def get_neighbors_within_distance(
        self, agent, distance, center=False, relation="intersects"
    ):
        """Return a list of agents within `distance` of `agent`.

        Distance is measured as a buffer around the agent's geometry,
        set center=True to calculate distance from center.
        """
        yield from self._agent_layer.get_neighbors_within_distance(
            agent, distance, center, relation
        )

    def agents_at(self, pos):
        """
        Return a list of agents at given pos.
        """
        return self._agent_layer.agents_at(pos)

    def distance(self, agent_a, agent_b):
        """
        Return distance of two agents.
        """
        return self._agent_layer.distance(agent_a, agent_b)

    def get_neighbors(self, agent):
        """
        Get (touching) neighbors of an agent.
        """
        return self._agent_layer.get_neighbors(agent)

    def get_agents_as_GeoDataFrame(self, agent_cls=GeoAgent) -> gpd.GeoDataFrame:
        """
        Extract GeoAgents as a GeoDataFrame.

        :param agent_cls: The class of the GeoAgents to extract. Default is `GeoAgent`.
        :return: A GeoDataFrame of the GeoAgents.
        :rtype: gpd.GeoDataFrame
        """

        return self._agent_layer.get_agents_as_GeoDataFrame(agent_cls)


class _AgentLayer:
    """
    Layer that contains the GeoAgents. Mainly for internal usage within `GeoSpace`.
    """

    def __init__(self):
        # neighborhood graph for touching neighbors
        self._neighborhood = None
        # rtree index for spatial indexing (e.g., neighbors within distance, agents at pos, etc.)
        self._idx = None
        self._id_to_agent = {}
        # bounds of the layer in [min_x, min_y, max_x, max_y] format
        # While it is possible to calculate the bounds from rtree index,
        # total_bounds is almost always needed (e.g., for plotting), while rtree index is not.
        # Hence we compute total_bounds separately from rtree index.
        self._total_bounds = None

    @property
    def agents(self):
        """
        Return a list of all agents in the layer.
        """

        return list(self._id_to_agent.values())

    @property
    def total_bounds(self):
        """
        Return the bounds of the layer in [min_x, min_y, max_x, max_y] format.
        """

        if self._total_bounds is None and len(self.agents) > 0:
            bounds = np.array([agent.geometry.bounds for agent in self.agents])
            min_x, min_y = np.min(bounds[:, :2], axis=0)
            max_x, max_y = np.max(bounds[:, 2:], axis=0)
            self._total_bounds = np.array([min_x, min_y, max_x, max_y])
        return self._total_bounds

    def _invalidate(self):
        """
        Invalidate the rtree index and neighborhood graph.
        """

        self._idx = None
        self._neighborhood = None
        self._total_bounds = None

    def _get_rtree_intersections(self, geometry):
        """
        Calculate rtree intersections for candidate agents.
        """

        self._ensure_index()
        if self._idx is None:
            return []
        else:
            return [
                self._id_to_agent[i] for i in self._idx.intersection(geometry.bounds)
            ]

    def _create_neighborhood(self):
        """
        Create a neighborhood graph of all agents.
        """

        agents = self.agents
        geometries = [agent.geometry for agent in agents]
        self._neighborhood = weights.contiguity.Queen.from_iterable(geometries)
        self._neighborhood.agents = agents
        self._neighborhood.idx = {}
        for agent, key in zip(agents, self._neighborhood.neighbors.keys()):
            self._neighborhood.idx[agent] = key

    def _ensure_index(self):
        """
        Ensure that the rtree index is created.
        """

        if self._idx is None:
            self._recreate_rtree()

    def _recreate_rtree(self, new_agents=None):
        """
        Create a new rtree index from agents geometries.
        """

        if new_agents is None:
            new_agents = []
        agents = list(self.agents) + new_agents

        if len(agents) > 0:
            # Bulk insert agents
            index_data = (
                (agent.unique_id, agent.geometry.bounds, None) for agent in agents
            )
            self._idx = index.Index(index_data)

    def add_agents(self, agents):
        """
        Add a list of GeoAgents to the layer without checking their crs.

        GeoAgents must have the same crs to avoid incorrect spatial indexing results.
        To change the crs of a GeoAgent, use `GeoAgent.to_crs()` method. Refer to
        `GeoSpace._check_agent()` as an example.
        This function may also be called with a single GeoAgent.

        :param agents: A list of GeoAgents or a single GeoAgent to be added into the layer.
        """

        if isinstance(agents, GeoAgent):
            agent = agents
            self._id_to_agent[agent.unique_id] = agent
            if self._idx:
                self._idx.insert(agent.unique_id, agent.geometry.bounds, None)
        else:
            for agent in agents:
                self._id_to_agent[agent.unique_id] = agent
            if self._idx:
                self._recreate_rtree(agents)
        self._total_bounds = None

    def remove_agent(self, agent):
        """
        Remove an agent from the layer.
        """

        del self._id_to_agent[agent.unique_id]
        if self._idx:
            self._idx.delete(agent.unique_id, agent.geometry.bounds)
        self._total_bounds = None

    def get_relation(self, agent, relation):
        """Return a list of related agents.

        Args:
            agent: the agent for which to compute the relation
            relation: must be one of 'intersects', 'within', 'contains',
                'touches'
            other_agents: A list of agents to compare against.
                Omit to compare against all other agents of the layer.
        """

        self._ensure_index()
        possible_agents = self._get_rtree_intersections(agent.geometry)
        for other_agent in possible_agents:
            if (
                getattr(agent.geometry, relation)(other_agent.geometry)
                and other_agent.unique_id != agent.unique_id
            ):
                yield other_agent

    def get_intersecting_agents(self, agent):
        self._ensure_index()
        intersecting_agents = self.get_relation(agent, "intersects")
        return intersecting_agents

    def get_neighbors_within_distance(
        self, agent, distance, center=False, relation="intersects"
    ):
        """Return a list of agents within `distance` of `agent`.

        Distance is measured as a buffer around the agent's geometry,
        set center=True to calculate distance from center.
        """
        self._ensure_index()
        if center:
            geometry = agent.geometry.centroid.buffer(distance)
        else:
            geometry = agent.geometry.buffer(distance)
        possible_neighbors = self._get_rtree_intersections(geometry)
        prepared_geometry = prep(geometry)
        for other_agent in possible_neighbors:
            if getattr(prepared_geometry, relation)(other_agent.geometry):
                yield other_agent

    def agents_at(self, pos):
        """
        Return a generator of agents at given pos.
        """

        self._ensure_index()
        if not isinstance(pos, Point):
            pos = Point(pos)

        possible_agents = self._get_rtree_intersections(pos)
        for other_agent in possible_agents:
            if pos.within(other_agent.geometry):
                yield other_agent

    def distance(self, agent_a, agent_b):
        """
        Return distance of two agents.
        """

        return agent_a.geometry.distance(agent_b.geometry)

    def get_neighbors(self, agent):
        """
        Get (touching) neighbors of an agent.
        """

        if not self._neighborhood or self._neighborhood.agents != self.agents:
            self._create_neighborhood()

        if self._neighborhood is None:
            return []
        else:
            idx = self._neighborhood.idx[agent]
            neighbors_idx = self._neighborhood.neighbors[idx]
            neighbors = [self.agents[i] for i in neighbors_idx]
            return neighbors

    def get_agents_as_GeoDataFrame(self, agent_cls=GeoAgent) -> gpd.GeoDataFrame:
        """
        Extract GeoAgents as a GeoDataFrame.

        :param agent_cls: The class of the GeoAgents to extract. Default is `GeoAgent`.
        :return: A GeoDataFrame of the GeoAgents.
        :rtype: geopandas.GeoDataFrame
        """

        agents_list = []
        crs = None
        for agent in self.agents:
            if isinstance(agent, agent_cls):
                crs = agent.crs
                agent_dict = {
                    attr: value
                    for attr, value in vars(agent).items()
                    if attr not in {"model", "pos", "_crs"}
                }
                agents_list.append(agent_dict)
        agents_gdf = gpd.GeoDataFrame.from_records(agents_list)
        # workaround for geometry column not being set in `from_records`
        # see https://github.com/geopandas/geopandas/issues/3152
        # may be removed when the issue is resolved
        agents_gdf.set_geometry("geometry", inplace=True)
        agents_gdf.crs = crs
        return agents_gdf
