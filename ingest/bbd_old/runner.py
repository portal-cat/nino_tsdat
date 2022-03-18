import os
from glob import glob

from ingest.bbd import Pipeline
from utils import expand, set_env

if __name__ == "__main__":
    set_env()
    pipeline = Pipeline(
        expand("config/pipeline_config_bbd.yml", __file__),
        expand("config/storage_config_bbd.yml", __file__),
    )
    pipeline.run(expand("tests/data/input/data_Jan_one_day.csv", __file__))
