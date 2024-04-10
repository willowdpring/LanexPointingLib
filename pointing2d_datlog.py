# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 15:21:22 2022

@author: willo

A class for logging arbitrary data to sequentialy nested dicts in a json like structure

"""

import os 
import json

class Logger:
    def __init__(self):
        self.variables = {}
        self.categories = {}

    def log_variable(self, name, value):
        if hasattr(value, '__dict__'):
            self.variables[name] = self._convert_to_dict(value)
        elif hasattr(value, 'tolist'):
            self.variables[name] = value.tolist() # np arrays are non serialisable
        else:
            self.variables[name] = value

    def add_category(self, category_name):
        self.categories[category_name] = Logger()

    def log_variable_to_category(self, category_name, name, value):
        if category_name not in self.categories:
            self.add_category(category_name)
        self.categories[category_name].log_variable(name, value)

    def _convert_to_dict(self, obj):
        if isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        elif isinstance(obj, (list, tuple)):
            return [self._convert_to_dict(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: self._convert_to_dict(value) for key, value in obj.items()}
        elif hasattr(obj, '__dict__'):
            return {attr: self._convert_to_dict(getattr(obj, attr)) for attr in obj.__dict__}
        else:
            return str(obj)

    def to_dict(self):
        result = {'variables': self.variables}
        for category_name, category_logger in self.categories.items():
            result[category_name] = category_logger.to_dict()
        return result
    
    def save_to_file(self, fname):
        filepath = '\\'.join(os.path.abspath(fname).split('\\')[:-1])
        if not os.path.exists(filepath):
            os.mkdir(filepath)
        try:
            with open(fname, 'w') as f:
                json.dump(self.to_dict(), f)
        except:
            print(self.to_dict())
            raise TypeError
