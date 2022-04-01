import yaml

class Config(object):  

   def __init__(self, config_path):
      with open(config_path) as cf_file:
         self._data = yaml.safe_load( cf_file.read() )

   def get(self, data_item):
      dictionary = dict(self._data)
      return dictionary.get(data_item)