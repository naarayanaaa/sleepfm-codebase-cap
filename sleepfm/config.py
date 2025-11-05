"""Sets paths based on configuration files."""

import configparser
import os
import types

_FILENAME = None
_PARAM = {}

CONFIG = types.SimpleNamespace(
    FILENAME=_FILENAME,
    DATASETS=_PARAM.get("datasets", "datasets"),
    OUTPUT=_PARAM.get("output", "output"),
    CACHE=_PARAM.get("cache", ".cache"),
)

#define the paths
PATH_TO_RAW_DATA = r"C:\Users\JD\sleepfm-codebase-cap\data_rbd_only\raw"
PATH_TO_PROCESSED_DATA = r"C:\Users\JD\sleepfm-codebase-cap\data_rbd_only\processed"

# Define Sleep related global variables

LABELS_DICT = {
    "Wake": 0, 
    "Stage 1": 1, 
    "Stage 2": 2, 
    "Stage 3": 3, 
    "REM": 4
}

MODALITY_TYPES = ["respiratory", "sleep_stages", "ekg"]
CLASS_LABELS = ["Wake", "Stage 1", "Stage 2", "Stage 3", "REM"]
NUM_CLASSES = 5

EVENT_TO_ID = {
    "Wake": 1, 
     "Stage 1": 2, 
     "Stage 2": 3, 
     "Stage 3": 4, 
     "Stage 4": 4, 
     "REM": 5,
}

LABEL_MAP = {
    "Sleep stage W": "Wake", 
    "Sleep stage N1": "Stage 1", 
    "Sleep stage N2": "Stage 2", 
    "Sleep stage N3": "Stage 3", 
    "Sleep stage R": "REM", 
    "W": "Wake", 
    "N1": "Stage 1", 
    "N2": "Stage 2", 
    "N3": "Stage 3", 
    "REM": "REM", 
    "wake": "Wake", 
    "nonrem1": "Stage 1", 
    "nonrem2": "Stage 2", 
    "nonrem3": "Stage 3", 
    "rem": "REM", 
}


# Define the channels in your dataset
ALL_CHANNELS = [
    'F4-C4', 'C4-P4', 'P4-O2', 'ROC-LOC', 'EMG1-EM2', 'ECG1-ECG2', 'C4-A1', 'SX1-SX2'
]


CHANNEL_DATA = {
    "Respiratory": [],  # none for this test run
    "Sleep_Stages": ['F4-C4','C4-P4','P4-O2','ROC-LOC','EMG1-EM2'],
    "EKG": ['ECG1-ECG2'],
}

CHANNEL_DATA_IDS = {
    "Respiratory": [ALL_CHANNELS.index(c) for c in CHANNEL_DATA["Respiratory"]],
    "Sleep_Stages": [ALL_CHANNELS.index(c) for c in CHANNEL_DATA["Sleep_Stages"]],
    "EKG": [ALL_CHANNELS.index(c) for c in CHANNEL_DATA["EKG"]],
}
