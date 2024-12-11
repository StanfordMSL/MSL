"""
Helper functions for handling parameters data.
"""

import os
import json

from typing import Dict, List, Union 

class Configs:
    """
    Class to handle the configuration of parameters data.
    """
    # Pre-define the configs dictionary format
    configs: Dict[str, Union[Dict[str,Union[str, float, List[float]]], str]]

    def __init__(self, fout_config:str=None, 
                 mpc_config:str='flightroom', 
                 drone_config:str='carl',
                 control_config:str='body_rate_v1',
                 default_config_path:str=None):
        """
        Initialize the Configs class to hold the various params used. By
        default, the class is initialized with the configurations for the
        fout waypoints, the MPC parameters, the drone parameters, and the
        control parameters.
        
        Args:
            - fout_config:         Name of the JSON file containing the fout waypoints.
            - mpc_config:          Name of the JSON file containing the MPC parameters.
            - drone_config:        Name of the JSON file containing the drone parameters.
            - control_config:      Name of the JSON file containing the control parameters.
            - default_config_path: Path to the directory containing the JSON files.
        """

        # Set the configuration directory
        if default_config_path is None:
            default_workspace_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            self.default_config_path = os.path.join(default_workspace_path, 'configs')
        else:
            self.default_config_path = default_config_path
        
        # Initialize the configs dictionary
        self.configs = {}

        # Add the default configurations
        if fout_config is not None:
            self.add_config('fout_waypoints', fout_config)
            self.add_config('mpc_parameters', mpc_config)
            self.add_config('drone_parameters', drone_config)
            self.add_config('control_parameters', control_config)

    def add_config(self, key, value, config_path=None):
        """
        Add a configuration to the dictionary. The configuration is loaded
        from a JSON file in the specified directory (default_config path if
        not specified).

        Args:
            - key:         Key to store the configuration under.
            - value:       Name of the JSON file to load the configuration from.
            - config_path: Path to the directory containing the JSON file.
        """

        # Check if the key exists as one of the folders in the config path
        if config_path is None:
            config_path = self.default_config_path

        key_path = os.path.join(config_path, key)
        if not os.path.isdir(key_path):
            raise ValueError(f"The key '{key}' does not correspond to a valid directory in the config path.")
        
        # Load the dictionary from the JSON file in the directory
        json_file_path = os.path.join(key_path, value + '.json')
        if not os.path.isfile(json_file_path):
            raise ValueError(f"The file '{json_file_path}' does not exist.")
        
        # Check if the key already exists
        if key in self.configs:
            print(f"Warning: The key '{key}' already exists. Overwriting the existing value.")
        
        # Add the configuration to the dictionary
        with open(json_file_path, 'r') as json_file:
            self.configs[key] = {
                'parameters':json.load(json_file),
                'name':value,'path':json_file_path}

    def remove_config(self, key):
        """
        Remove a configuration from the dictionary.

        Args:
            - key:         Key to remove from the dictionary.

        Returns:
            - config:      The configuration that was removed.
        """

        return self.configs.pop(key, None)

    def list_configs(self):
        """
        List the configurations currently loaded in the dictionary.
        """
        configs_list = list(self.configs.keys())
        Nstr = max([len(config) for config in configs_list])

        print("Currently Loaded Configs:")
        for config in configs_list:
            param_category = config.ljust(Nstr)
            param_name = self.configs[config]['name']
            print(f" - {param_category}: {param_name}")

    def get_params(self, key):
        """
        Get the parameters from the configuration dictionary.
        
        Args:
            - key:         Key to get the parameters for.

        Returns:
            - parameters:  Dictionary containing the parameters
        """

        return self.configs.get(key).get('parameters')

    def __str__(self):
        """ 
        Return a string representation of the Configs object

        Returns:
            - str:         String representation of the Configs object
        """
        return str(self.configs)