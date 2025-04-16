import h5py
import numpy as np
import os


class DatasetWriter:
    def __init__(self) -> None:
        pass

    def write_dataset(data: dict, path: str):
        """Writes HDF5 file from data. Non array data written to "scalar_data" dataset as attributes.

        :param data: dictionary containing data to be written.
            Object format:

            data: {
                [dataset_name]: {
                    dataset_data: np.ndarray | int | float,
                    dataset_attributes: {[attribute_name]: str}
                }
            }

            Example:

            ```python
            data = {
                "my_dataset_1": {
                    "dataset_data": np.array([1, 2, 3]),
                    "dataset_attributes": {
                        "description": "small numbers",
                        "location": "collected at lab X",
                    },
                },
                "my_dataset_2": {
                    "dataset_data": np.array([3, 2, 1]),
                    "dataset_attributes": {
                        "my_attribute_1": "small numbers",
                        "my_attribute_2": "collected at lab Y",
                    },
                },
            }
            ```
        :type data: dict[str, dict[str, np.ndarray  |  dict]]
        :param path: Where the file will be saved.
        :type path: str
        """

        # Create file
        dir = os.path.dirname(path)
        if not os.path.exists(dir):
            os.makedirs(dir)
        file = h5py.File(path, "w")
        # Create datasets
        scalar_data_dataset = file.create_dataset("scalar_data", (1,), dtype="f")
        for key, value in data.items():

            dataset_name = key
            dataset_properties = value
            dataset_data = dataset_properties["dataset_data"]
            dataset_attributes = dataset_properties["dataset_attributes"]

            if isinstance(dataset_data, np.ndarray):
                dataset = file.create_dataset(name=dataset_name, data=dataset_data)
                dataset.attrs.update(dataset_attributes)
            else:
                scalar_data_dataset.attrs[dataset_name] = str(dataset_data)

        file.close()
