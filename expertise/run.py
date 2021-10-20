import argparse
import json
from pathlib import Path

from .execute_expertise import execute_expertise

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="a JSON file containing all other arguments")
    args = parser.parse_args()

    config_file_path = Path(args.config)
    with open(config_file_path) as file_handle:
        config = json.load(file_handle)
    execute_expertise(config)
