# todo: create the pipeline for end to end with image classification
# I can use the code from the regular end to end pipe and from the playground
# psudo code:
"""
- Perform trichome detection using the faster rcnn model or yolo model
- Filter Predictions
- Save the filtered predictions
    - Filter large objects
    - Filter blurry objects using the image classification model.
- Perform Classification using the image classification model.
- Save and plot the results if needed
"""