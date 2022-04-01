import preprocess as ps 
import ffnn as ffnn 
from config import Config
import run_ffnn 

def main(yaml_file_name):
  cfg = Config(yaml_file_name)
  run_ffnn.full_model(cfg)

if __name__ == "__main__":
  #config_file = '../yaml/config.yaml'
  config_file = '../yaml/synthetic.yaml'
  main(config_file)