"""
config.py

This module handles the configuration settings for the Image Background Remover application. 
It loads the configuration from a JSON file located at '/etc/config.json' and provides access 
to the SECRET_KEY for secure operations within the application.
"""

import json

with open('/etc/config.json') as config_file:
	config = json.load(config_file)

class Config:
	SECRET_KEY = config.get('SECRET_KEY')

