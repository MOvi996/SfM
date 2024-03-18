from sfm import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, default="config.json")
args = parser.parse_args()

model = SFM(args.config_path)
model.run()