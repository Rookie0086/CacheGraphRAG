import os

_DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")

RGB_DATAPATH = os.path.join(_DATA_ROOT, "rgb")
HOTPOTQA_DATAPATH = os.path.join(_DATA_ROOT, "hotpotqa")
SPECIFICQA_DATAPATH = os.path.join(_DATA_ROOT, "specificqa")
TWOWIKIMULTIHOPQA_DATAPATH = os.path.join(_DATA_ROOT, "2wikimultihopqa")
