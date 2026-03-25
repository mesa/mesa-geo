import numpy as np


class PropertyLayer:
    """
    PropertyLayer manages raster attributes and cell data separately
    from RasterLayer to improve modularity and maintainability.
    """

    def __init__(self, width, height, raster_layer):
        self.width = width
        self.height = height
        self.raster_layer = raster_layer

        # store attribute names
        self.attributes = set()

        # store raster data
        self.data = {}

    def apply_raster(self, data, attr_name=None):
        if data.ndim == 2:
            data = data[np.newaxis, ...]

        num_bands = data.shape[0]

        if attr_name is None:
            attr_names = [f"attribute_{i}" for i in range(num_bands)]
        elif isinstance(attr_name, str):
            attr_names = [f"{attr_name}_{i}" for i in range(num_bands)]
        else:
            attr_names = attr_name

        for band_idx, attr in enumerate(attr_names):
            self.attributes.add(attr)

            for grid_x in range(self.width):
                for grid_y in range(self.height):
                    setattr(
                        self.raster_layer.cells[grid_x][grid_y],
                        attr,
                        data[band_idx, self.height - grid_y - 1, grid_x],
                    )

            self.data[attr] = data[band_idx]

    def get_raster(self, attr_name=None):
        if attr_name is None:
            return self.data

        if isinstance(attr_name, str):
            return self.data[attr_name]

        return {name: self.data[name] for name in attr_name}
