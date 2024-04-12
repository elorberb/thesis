import os
import shutil
import random
from pathlib import Path
from IPython import display as ipydisplay
import ultralytics
from ultralytics.data.converter import convert_coco
import config
import shutil

   
def validate_version_and_gpu():
    """
    Clears the current IPython output and checks the ultralytics package version and GPU availability.
    """
    # Clear the current IPython output to ensure the display is not cluttered
    ipydisplay.clear_output(wait=True)
    
    # Perform the checks using ultralytics package
    ultralytics.checks()
    