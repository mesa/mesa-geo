import random
import unittest
import warnings

import geopandas as gpd
import mesa
import numpy as np
from shapely.geometry import Point, Polygon

import mesa_geo as mg


class TestGeoSpace(unittest.TestCase):
    def setUp(self) -> None:
        self.model = mesa.Model()
        self.model.space = mg.GeoSpace(crs="epsg:4326")
        self.agent_creator = mg.AgentCreator(
            agent_class=mg.GeoAgent, model=self.model, crs="epsg:3857"
        )
        self.geometries = [Point(1, 1)] * 7
        self.agents = [
            self.agent_creator.create_agent(
                geometry=geometry,
            )
            for geometry in self.geometries
        ]
        self.polygon_agent = mg.GeoAgent(
            model=self.model,
            geometry=Polygon([(0, 0), (0, 2), (2, 2), (2, 0)]),
            crs="epsg:3857",
        )
        self.touching_agent = mg.GeoAgent(
            model=self.model,
            geometry=Polygon([(2, 0), (2, 2), (4, 2), (4, 0)]),
            crs="epsg:3857",
        )
        self.disjoint_agent = mg.GeoAgent(
            model=self.model,
            geometry=Polygon([(10, 10), (10, 12), (12, 12), (12, 10)]),
            crs="epsg:3857",
        )
        self.image_layer = mg.ImageLayer(
            values=np.random.uniform(low=0, high=255, size=(3, 500, 500)),
            crs="epsg:4326",
            total_bounds=[
                -122.26638888878,
                42.855833333,
                -121.94972222209202,
                43.01472222189958,
            ],
        )
        self.vector_layer = gpd.GeoDataFrame(
            {"name": ["point_1", "point_2"], "geometry": [Point(1, 2), Point(2, 1)]},
            crs="epsg:4326",
        )
        self.geo_space = mg.GeoSpace()
        self.geo_space_with_different_crs = mg.GeoSpace(crs="epsg:2283")

    def tearDown(self) -> None:
        pass

    def test_add_agents_with_the_same_crs(self):
        self.assertEqual(len(self.geo_space.agents), 0)
        self.geo_space.add_agents(self.agents)
        self.assertEqual(len(self.geo_space.agents), len(self.agents))

        for agent in self.geo_space.agents:
            self.assertEqual(agent.crs, self.geo_space.crs)

    def test_add_agents_with_different_crs(self):
        self.assertEqual(len(self.geo_space_with_different_crs.agents), 0)
        with self.assertWarns(Warning):
            self.geo_space_with_different_crs.add_agents(self.agents)

        self.geo_space_with_different_crs.warn_crs_conversion = False
        # assert no warning
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self.geo_space_with_different_crs.add_agents(self.agents)

        self.geo_space_with_different_crs.add_agents(self.agents)
        self.assertEqual(
            len(self.geo_space_with_different_crs.agents), len(self.agents)
        )

        for agent in self.geo_space_with_different_crs.agents:
            self.assertEqual(agent.crs, self.geo_space_with_different_crs.crs)

    def test_remove_agent(self):
        self.geo_space.add_agents(self.agents)
        agent_to_remove = random.choice(self.agents)
        self.geo_space.remove_agent(agent_to_remove)
        remaining_agent_idx = {agent.unique_id for agent in self.geo_space.agents}

        self.assertEqual(len(self.geo_space.agents), len(self.agents) - 1)
        self.assertTrue(agent_to_remove.unique_id not in remaining_agent_idx)

    def test_add_image_layer(self):
        with self.assertWarns(Warning):
            self.geo_space.add_layer(self.image_layer)
        self.assertEqual(len(self.geo_space.layers), 1)

        self.geo_space.warn_crs_conversion = False
        # assert no warning
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self.geo_space.add_layer(self.image_layer)
        self.assertEqual(len(self.geo_space.layers), 2)

    def test_add_vector_layer(self):
        with self.assertWarns(Warning):
            self.geo_space.add_layer(self.vector_layer)
        self.assertEqual(len(self.geo_space.layers), 1)

        self.geo_space.warn_crs_conversion = False
        # assert no warning
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self.geo_space.add_layer(self.vector_layer)
        self.assertEqual(len(self.geo_space.layers), 2)

    def test_get_neighbors_within_distance(self):
        self.geo_space.add_agents(self.agents)
        agent_to_check = random.choice(self.agents)

        neighbors = list(
            self.geo_space.get_neighbors_within_distance(
                agent_to_check, distance=1.0, center=True
            )
        )
        self.assertEqual(len(neighbors), 7)

        neighbors = list(
            self.geo_space.get_neighbors_within_distance(agent_to_check, distance=1.0)
        )
        self.assertEqual(len(neighbors), 7)

    def test_get_agents_as_GeoDataFrame(self):
        self.geo_space.add_agents(self.agents)

        agents_list = [{"geometry": agent.geometry} for agent in self.agents]
        agents_gdf = gpd.GeoDataFrame.from_records(agents_list)
        # workaround for geometry column not being set in `from_records`
        # see https://github.com/geopandas/geopandas/issues/3152
        # may be removed when the issue is resolved
        agents_gdf.set_geometry("geometry", inplace=True)
        agents_gdf.crs = self.geo_space.crs

        self.assertEqual(
            self.geo_space.get_agents_as_GeoDataFrame().crs, agents_gdf.crs
        )

    def test_get_relation_contains(self):
        self.geo_space.add_agents(self.polygon_agent)
        self.assertEqual(
            list(self.geo_space.get_relation(self.polygon_agent, relation="contains")),
            [],
        )

        self.geo_space.add_agents(self.agents)
        agents_id = {agent.unique_id for agent in self.agents}
        contained_agents_id = {
            agent.unique_id
            for agent in self.geo_space.get_relation(
                self.polygon_agent, relation="contains"
            )
        }
        self.assertEqual(contained_agents_id, agents_id)

    def test_get_relation_within(self):
        self.geo_space.add_agents(self.agents[0])
        self.assertEqual(
            list(self.geo_space.get_relation(self.agents[0], relation="within")), []
        )
        self.geo_space.add_agents(self.polygon_agent)
        within_agent = next(
            self.geo_space.get_relation(self.agents[0], relation="within")
        )
        self.assertEqual(within_agent.unique_id, self.polygon_agent.unique_id)

    def test_get_relation_touches(self):
        self.geo_space.add_agents(self.polygon_agent)
        self.assertEqual(
            list(self.geo_space.get_relation(self.polygon_agent, relation="touches")),
            [],
        )
        self.geo_space.add_agents(self.touching_agent)
        self.assertEqual(
            len(
                list(
                    self.geo_space.get_relation(self.polygon_agent, relation="touches")
                )
            ),
            1,
        )
        self.assertEqual(
            next(
                self.geo_space.get_relation(self.polygon_agent, relation="touches")
            ).unique_id,
            self.touching_agent.unique_id,
        )

    def test_get_relation_intersects(self):
        self.geo_space.add_agents(self.polygon_agent)
        self.assertEqual(
            list(
                self.geo_space.get_relation(self.polygon_agent, relation="intersects")
            ),
            [],
        )

        self.geo_space.add_agents(self.agents)
        agents_id = {agent.unique_id for agent in self.agents}
        intersecting_agents_id = {
            agent.unique_id
            for agent in self.geo_space.get_relation(
                self.polygon_agent, relation="intersects"
            )
        }
        self.assertEqual(intersecting_agents_id, agents_id)

        # disjoint agent should not be returned since it is not intersecting
        self.geo_space.add_agents(self.disjoint_agent)
        intersecting_agents_id = {
            agent.unique_id
            for agent in self.geo_space.get_relation(
                self.polygon_agent, relation="intersects"
            )
        }
        self.assertEqual(intersecting_agents_id, agents_id)

    def test_get_intersecting_agents(self):
        self.geo_space.add_agents(self.polygon_agent)
        self.assertEqual(
            list(self.geo_space.get_intersecting_agents(self.polygon_agent)),
            [],
        )

        self.geo_space.add_agents(self.agents)
        agents_id = {agent.unique_id for agent in self.agents}
        intersecting_agents_id = {
            agent.unique_id
            for agent in self.geo_space.get_intersecting_agents(self.polygon_agent)
        }
        self.assertEqual(intersecting_agents_id, agents_id)

        # disjoint agent should not be returned since it is not intersecting
        self.geo_space.add_agents(self.disjoint_agent)
        intersecting_agents_id = {
            agent.unique_id
            for agent in self.geo_space.get_intersecting_agents(self.polygon_agent)
        }
        self.assertEqual(intersecting_agents_id, agents_id)

    def test_agents_at(self):
        self.geo_space.add_agents(self.agents)
        self.assertEqual(
            len(list(self.geo_space.agents_at(self.agents[0].geometry))),
            len(self.agents),
        )
        agents_id = {agent.unique_id for agent in self.agents}
        agents_id_found = {
            agent.unique_id for agent in self.geo_space.agents_at((1, 1))
        }
        self.assertEqual(agents_id_found, agents_id)

    def test_get_neighbors(self):
        self.geo_space.add_agents(self.polygon_agent)
        self.assertEqual(len(self.geo_space.get_neighbors(self.polygon_agent)), 0)
        self.geo_space.add_agents(self.touching_agent)
        self.assertEqual(len(self.geo_space.get_neighbors(self.polygon_agent)), 1)
        self.assertEqual(
            self.geo_space.get_neighbors(self.polygon_agent)[0].unique_id,
            self.touching_agent.unique_id,
        )

    def test_to_crs_not_inplace(self):
        model = mesa.Model()
        space = mg.GeoSpace(crs="EPSG:3395")

        agent = mg.GeoAgent(model=model, geometry=Point(0, 0), crs="EPSG:3395")
        space.add_agents([agent])

        new_space = space.to_crs("EPSG:3857", inplace=False)

        self.assertEqual(space.crs.to_epsg(), 3395)
        self.assertEqual(space.transformer.source_crs.to_epsg(), space.crs.to_epsg())
        self.assertEqual(new_space.crs.to_epsg(), 3857)
        self.assertEqual(
            new_space.transformer.source_crs.to_epsg(), new_space.crs.to_epsg()
        )

        self.assertEqual(space.agents[0].crs.to_epsg(), 3395)
        new_agent = new_space.agents[0]
        self.assertIsNot(new_agent, agent)
        self.assertEqual(new_agent.crs.to_epsg(), 3857)

    def test_to_crs_inplace(self):
        model = mesa.Model()
        space = mg.GeoSpace(crs="EPSG:3395")

        agent = mg.GeoAgent(model=model, geometry=Point(0, 0), crs="EPSG:3395")
        space.add_agents([agent])

        space.to_crs("EPSG:3857", inplace=True)

        self.assertEqual(space.crs.to_epsg(), 3857)
        self.assertEqual(space.agents[0].crs.to_epsg(), 3857)
        self.assertEqual(space.transformer.source_crs.to_epsg(), space.crs.to_epsg())
