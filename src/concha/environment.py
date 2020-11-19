import os
import json
from pathlib import Path


class FileHandler:
    """Filehandler automates some json -> file and vice versa operations."""

    def __init__(self):
        """Figure out in what sort of environment the concha Planner is running."""

        # Detect if concha is running in a colab notebook with a Google Drive attached
        # And if so, whether a concha_planners folder already exists
        is_colab = os.path.exists(os.path.join("/content", "sample_data"))
        is_google_drive_mounted = os.path.exists(
            os.path.join("/content", "drive", "My Drive")
        )

        colab_drive_path = os.path.join("/content", "drive", "My Drive")
        colab_path = "/content"
        home_path = Path.home()

        if is_colab and is_google_drive_mounted:
            self.base_path = colab_drive_path
        elif is_colab:
            self.base_path = colab_path
        else:
            self.base_path = home_path

    def check_planner_path(self, planner_name):
        """Make sure the necessary directories exist  for the planner.

        Args:
            planner_name (str): The name of the planner within the concha_planners/ directory.

        Returns:
            planner_path (str): The path for the planner (top level)
            settings_path (str): The path for the planner_settings.json file
        """
        planner_path = os.path.join(self.base_path, "concha_planners", planner_name)
        subdirs = ["history", "metadata", "forecast", "models"]
        for subdir in subdirs:
            subpath = os.path.join(planner_path, subdir)
            if not os.path.exists(subpath):
                os.makedirs(subpath)
        settings_path = os.path.join(planner_path, "planner_settings.json")
        return planner_path, settings_path

    def check_importer_path(self):
        """Make sure the importer path (the folder for saving the importer settings) exists.

        Returns:
            importers_path (str): Path to ...concha_planners/importers/
        """

        importers_path = os.path.join(self.base_path, "concha_planners", "importers")
        if not os.path.exists(importers_path):
            os.makedirs(importers_path)
        return importers_path

    def dict_to_file(self, dct, file_path):
        """Saves a dict to a json file at the specified path.

        Args:
            dct (dict): The dict to save
            file_path (str): The path at which to save the dict as a json

        Returns:
            None
        """
        with open(file_path, "w") as file:
            json.dump(dct, file, indent=4)
        return

    def dict_from_file(self, file_path):
        """Loads a dict from a json file specified at file_path.

        Args:
            file_path (str): Path from which to load .json file

        Returns:
            dct (dict): A dict of the object stored in json format at the file_path.
        """
        with open(file_path, "r") as file:
            dct = json.load(file)
        return dct
