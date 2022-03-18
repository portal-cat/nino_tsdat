import re

from typing import AnyStr, Dict
from utils import IngestSpec, expand
from . import Pipeline


mapping: Dict["AnyStr@compile", IngestSpec] = {
    # Mapping for Raw Data -> Ingest
    re.compile(r"_\d{4}.csv"): IngestSpec(
        pipeline=Pipeline,
        pipeline_config=expand("config/pipeline_config_bbd.yml", __file__),
        storage_config=expand("config/storage_config_bbd.yml", __file__),
        name="bbd",
    ),
    # Mapping for Processed Data -> Ingest (so we can reprocess plots)
    re.compile(r"_\d{4}.csv"): IngestSpec(
        pipeline=Pipeline,
        pipeline_config=expand("config/pipeline_config_bbd.yml", __file__),
        storage_config=expand("config/storage_config_bbd.yml", __file__),
        name="plot_bbd",
    ),
}
