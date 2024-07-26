import os
import json
import csv
import numpy as np

def get_class_distribution(json_data):
    return {
        "clear_count": np.round(json_data["class_distribution"].get("1", 0), 4),
        "cloudy_count": np.round(json_data["class_distribution"].get("2", 0), 4),
        "amber_count": np.round(json_data["class_distribution"].get("3", 0), 4),
        "clear_normalized": np.round(json_data["normalized_class_distribution"].get("1", 0), 4),
        "cloudy_normalized": np.round(json_data["normalized_class_distribution"].get("2", 0), 4),
        "amber_normalized": np.round(json_data["normalized_class_distribution"].get("3", 0), 4)
    }

def collect_data_from_json(base_path, day_folders):
    data_per_image = []
    data_per_folder = []
    
    for day_folder in day_folders:
        day_folder_path = os.path.join(base_path, day_folder)
        print(f"Processing {day_folder_path}...")

        for subfolder in ['greenhouse', 'lab']:
            subfolder_path = os.path.join(day_folder_path, subfolder)
            if os.path.exists(subfolder_path):
                for numbered_folder in os.listdir(subfolder_path):
                    numbered_folder_path = os.path.join(subfolder_path, numbered_folder)
                    if os.path.isdir(numbered_folder_path):
                        json_file_path = os.path.join(numbered_folder_path, "class_distribution.json")
                        if os.path.exists(json_file_path):
                            with open(json_file_path, 'r') as f:
                                json_data = json.load(f)
                                
                                # Collect data per folder
                                folder_entry = {
                                    "day": day_folder,
                                    "location": subfolder,
                                    "number": numbered_folder,
                                }
                                folder_entry.update(get_class_distribution(json_data))
                                data_per_folder.append(folder_entry)
                                
                                # Collect data per image
                                for image_folder in os.listdir(numbered_folder_path):
                                    image_folder_path = os.path.join(numbered_folder_path, image_folder)
                                    if os.path.isdir(image_folder_path) and image_folder.startswith("IMG_"):
                                        for img_file in os.listdir(image_folder_path):
                                            if img_file.endswith("_class_distribution.json"):
                                                img_json_file_path = os.path.join(image_folder_path, img_file)
                                                with open(img_json_file_path, 'r') as img_f:
                                                    img_json_data = json.load(img_f)
                                                    image_entry = {
                                                        "day": day_folder,
                                                        "location": subfolder,
                                                        "number": numbered_folder,
                                                        "image": image_folder,
                                                    }
                                                    image_entry.update(get_class_distribution(img_json_data))
                                                    data_per_image.append(image_entry)
                        else:
                            print(f"Missing file: {json_file_path}")
            else:
                print(f"Missing subfolder: {subfolder_path}")

    return data_per_image, data_per_folder

def save_data_to_csv(data, output_file):
    if not data:
        print(f"No data to save for {output_file}")
        return
    
    fieldnames = list(data[0].keys())
    
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in data:
            writer.writerow(row)

def validate_image_folders(base_path, day_folders):
    current_number = 1

    for day_folder in day_folders:
        day_folder_path = os.path.join(base_path, day_folder)
        print(f"Checking {day_folder_path}...")

        # Check for greenhouse and lab folders
        for subfolder in ['greenhouse', 'lab']:
            subfolder_path = os.path.join(day_folder_path, subfolder)
            if os.path.exists(subfolder_path):
                # Check numbered folders within the expected range for this day
                for i in range(current_number, current_number + 30):
                    numbered_folder_path = os.path.join(subfolder_path, str(i))
                    if not os.path.exists(numbered_folder_path):
                        print(f"Missing folder: {numbered_folder_path}")
                    else:
                        # Check if the numbered folder contains any files
                        if not any(os.path.isfile(os.path.join(numbered_folder_path, f)) for f in os.listdir(numbered_folder_path)):
                            print(f"Empty folder: {numbered_folder_path}")
            else:
                print(f"Missing subfolder: {subfolder_path}")
        
        
if __name__ == '__main__':
    # detect missing folders in the assessing cannabis maturity experiment
    # base_path = "/sise/shanigu-group/etaylor/assessing_cannabis_exp/images"
    day_folders = [
        "day_1_2024_05_30", "day_2_2024_06_03", "day_3_2024_06_06", "day_4_2024_06_10",
        "day_5_2024_06_13", "day_6_2024_06_17", "day_7_2024_06_20", "day_8_2024_06_24",
        "day_9_2024_06_27"
    ]
    # validate_image_folders(base_path, day_folders)

    base_path = "/sise/shanigu-group/etaylor/assessing_cannabis_exp/results"
    output_folder = "/sise/shanigu-group/etaylor/assessing_cannabis_exp/results/csv_results"
    os.makedirs(output_folder, exist_ok=True)
    data_per_image, data_per_folder = collect_data_from_json(base_path, day_folders)
    save_data_to_csv(data_per_image, os.path.join(output_folder, "collected_class_distribution_per_image.csv"))
    save_data_to_csv(data_per_folder, os.path.join(output_folder, "collected_class_distribution_per_folder.csv"))

    print("Data collection complete. CSV files saved as 'collected_class_distribution_per_image.csv' and 'collected_class_distribution_per_folder.csv'.")
