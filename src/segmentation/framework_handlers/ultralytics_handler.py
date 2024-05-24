from IPython import display as ipydisplay
import ultralytics


models = [
    "rtdetr-x.pt",
    "yolov5xu.pt",
    "yolov8x.pt",
    "yolov9c.pt"
]

def validate_version_and_gpu():
    """
    Clears the current IPython output and checks the ultralytics package version and GPU availability.
    """
    # Clear the current IPython output to ensure the display is not cluttered
    ipydisplay.clear_output(wait=True)
    
    # Perform the checks using ultralytics package
    ultralytics.checks()
    