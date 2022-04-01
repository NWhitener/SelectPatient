import pandas as pd 

class Config(object):  

   def __init__(self, config_path):
      with open(config_path) as cf_file:
         self._data = pd.read_csv(config_path)

   def get(self, data_item):
      dictionary = dict(self._data)
      return dictionary.get(data_item)