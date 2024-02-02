import base64
import pandas as pd
import config

def get_image_as_base64(path):
    with open(path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode()
    return f"data:image/jpeg;base64,{encoded}"

# Experiment utils
def load_random_image(reviewed_images):
    df = pd.read_csv(config.GOOD_QUALITY_IMAGES_CSV)
    available_images = df[~df['image_number'].isin(reviewed_images)]

    if not available_images.empty:
        random_row = available_images.sample(n=1).iloc[0]
        image_number = random_row['image_number']
        week, zoom_type = config.find_image_details(image_number)

        if week and zoom_type:
            image_path = config.get_raw_image_path(week, zoom_type) / f"{image_number}.JPG"
            return str(image_path), image_number
        else:
            return None, None
    else:
        return None, None