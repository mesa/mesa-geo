"""
GeoAgent and AgentCreator classes
---------------------------------
"""

from __future__ import annotations

import copy
import json

import geopandas as gpd
import numpy as np
import pyproj
from mesa import Agent, Model
from shapely.geometry import mapping
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform

from mesa_geo.geo_base import GeoBase


class GeoAgent(Agent, GeoBase):
    """
    Base class for a geo model agent.
    """

    def __init__(self, model, geometry, crs):
        """
        Create a new agent.

        :param model: The model the agent is in.
        :param geometry: A Shapely object representing the geometry of the agent.
        :param crs: The coordinate reference system of the geometry.
        """

        Agent.__init__(self, model)
        GeoBase.__init__(self, crs=crs)
        self.geometry = geometry

    @property
    def total_bounds(self) -> np.ndarray | None:
        if self.geometry is not None:
            return self.geometry.bounds
        else:
            return None

    def to_crs(self, crs, inplace=False) -> GeoAgent | None:
        super()._to_crs_check(crs)

        agent = self if inplace else copy.copy(self)

        target_crs = pyproj.CRS.from_user_input(crs)
        if not agent.crs.is_exact_same(target_crs):
            transformer = pyproj.Transformer.from_crs(
                crs_from=agent.crs, crs_to=target_crs, always_xy=True
            )
            agent.geometry = agent.get_transformed_geometry(transformer)
            agent.crs = target_crs

        if not inplace:
            return agent

    def get_transformed_geometry(self, transformer):
        """
        Return the transformed geometry given a transformer.
        """

        return transform(transformer.transform, self.geometry)

    def step(self):
        """
        Advance one step.
        """

    def __geo_interface__(self):
        """
        Return a GeoJSON Feature. Removes geometry from attributes.
        """

        properties = dict(vars(self))
        properties["model"] = str(self.model)
        geometry = properties.pop("geometry")
        geometry = transform(self.model.space.transformer.transform, geometry)

        return {
            "type": "Feature",
            "geometry": mapping(geometry),
            "properties": properties,
        }


class AgentCreator:
    """
    Create GeoAgents from files, GeoDataFrames, GeoJSON or Shapely objects.
    """

    def __init__(self, agent_class, model=None, crs=None, agent_kwargs=None):
        """
        Define the agent_class and required agent_kwargs.

        :param agent_class: The class of the agent to create.
        :param model: The model to create the agent in.
        :param crs: The coordinate reference system of the agent. Default to None,
            and the crs from the file/GeoDataFrame/GeoJSON will be used.
            Otherwise, geometries are converted into this crs automatically.
        :param agent_kwargs: Keyword arguments to pass to the agent_class.
        """

        self.agent_class = agent_class
        self.model = model
        self.crs = crs
        self.agent_kwargs = agent_kwargs if agent_kwargs else {}

    @property
    def crs(self):
        """
        Return the coordinate reference system of the GeoAgents.
        """

        return self._crs

    @crs.setter
    def crs(self, crs):
        """
        Set the coordinate reference system of the GeoAgents.
        """

        self._crs = pyproj.CRS.from_user_input(crs) if crs else None

    def create_agent(self, geometry):
        """
        Create a single agent from a geometry. Shape must be a valid Shapely object.

        :param geometry: The geometry of the agent.
        :return: The created agent.
        :rtype: self.agent_class
        """

        if not isinstance(geometry, BaseGeometry):
            raise TypeError("Geometry must be a Shapely Geometry")

        if not self.crs:
            raise TypeError(
                f"Unable to set CRS for {self.agent_class.__name__} due to empty CRS in {self.__class__.__name__}"
            )

        if not isinstance(self.model, Model):
            raise ValueError("Model must be a valid Mesa model object")

        new_agent = self.agent_class(
            model=self.model,
            geometry=geometry,
            crs=self.crs,
            **self.agent_kwargs,
        )

        return new_agent

    def from_GeoDataFrame(self, gdf, set_attributes=True):
        """
        Create a list of agents from a GeoDataFrame.

        :param gdf: The GeoDataFrame to create agents from.
        :param set_attributes: Set agent attributes from GeoDataFrame columns.
            Default True.
        """

        if self.crs:
            if gdf.crs:
                gdf.to_crs(self.crs, inplace=True)
            else:
                gdf.set_crs(self.crs, inplace=True)
        else:
            if gdf.crs:
                self.crs = gdf.crs
            else:
                raise TypeError(
                    f"Unable to set CRS for {self.agent_class.__name__} due to empty CRS in both "
                    f"{self.__class__.__name__} and {gdf.__class__.__name__}."
                )

        agents = []
        for _, row in gdf.iterrows():
            geometry = row[gdf.geometry.name]
            new_agent = self.create_agent(geometry=geometry)

            if set_attributes:
                for col in row.index:
                    if col != gdf.geometry.name:
                        setattr(new_agent, col, row[col])
            agents.append(new_agent)

        return agents

    def from_file(self, filename, set_attributes=True):
        """
        Create agents from vector data files (e.g. Shapefiles).

        :param filename: The vector data file to create agents from.
        :param set_attributes: Set agent attributes from GeoDataFrame columns. Default True.
        """

        gdf = gpd.read_file(filename)
        agents = self.from_GeoDataFrame(gdf, set_attributes=set_attributes)
        return agents

    def from_GeoJSON(
        self,
        GeoJSON,  # noqa: N803
        set_attributes=True,
    ):
        """
        Create agents from a GeoJSON object or string. CRS is set to epsg:4326.

        :param GeoJSON: The GeoJSON object or string to create agents from.
        :param set_attributes: Set agent attributes from GeoDataFrame columns. Default True.
        """

        gj = json.loads(GeoJSON) if isinstance(GeoJSON, str) else GeoJSON

        gdf = gpd.GeoDataFrame.from_features(gj)
        # epsg:4326 is the CRS for all GeoJSON: https://datatracker.ietf.org/doc/html/rfc7946#section-4
        gdf.crs = "epsg:4326"
        agents = self.from_GeoDataFrame(gdf, set_attributes=set_attributes)
        return agents
