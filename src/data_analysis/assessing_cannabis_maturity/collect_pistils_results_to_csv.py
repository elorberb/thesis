import os
import json
import csv
import numpy as np

def get_color_ratios(json_data):
    """
    Extract aggregated pistils ratios from JSON data.
    Expected keys (if already aggregated) are:
      - "num_pistils"
      - "average_green_ratio"
      - "average_orange_ratio"
    Otherwise, if the JSON is a list (per-pistil entries),
    compute the simple average.
    """
    if isinstance(json_data, dict) and "num_pistils" in json_data:
        return {
            "num_pistils": json_data.get("num_pistils", 0),
            "avg_green_ratio": np.round(json_data.get("average_green_ratio", 0), 4),
            "avg_orange_ratio": np.round(json_data.get("average_orange_ratio", 0), 4)
        }
    elif isinstance(json_data, list) and json_data:
        num = len(json_data)
        avg_green = np.round(np.mean([entry.get("green_ratio", 0) for entry in json_data]), 4)
        avg_orange = np.round(np.mean([entry.get("orange_ratio", 0) for entry in json_data]), 4)
        return {"num_pistils": num, "avg_green_ratio": avg_green, "avg_orange_ratio": avg_orange}
    else:
        return {"num_pistils": 0, "avg_green_ratio": 0, "avg_orange_ratio": 0}

def aggregate_folder_entries(image_entries):
    """
    Given a list of per-image aggregated results (each with keys:
    "num_pistils", "avg_green_ratio", and "avg_orange_ratio"),
    compute a weighted aggregate.
    
    The folder's total pistils is the sum of all per-image num_pistils.
    The folder's average ratios are computed as:
      weighted_avg = sum(num_pistils * ratio) / total_num_pistils.
    Also compute the normalized pistils count as:
      num_pistils_normalized = total_num_pistils / number_of_images.
    """
    total = sum(entry["num_pistils"] for entry in image_entries)
    count = len(image_entries)
    if total == 0 or count == 0:
        return {"num_pistils": 0, "avg_green_ratio": 0, "avg_orange_ratio": 0, "num_pistils_normalized": 0}
    avg_green = np.round(
        sum(entry["num_pistils"] * entry["avg_green_ratio"] for entry in image_entries) / total, 4
    )
    avg_orange = np.round(
        sum(entry["num_pistils"] * entry["avg_orange_ratio"] for entry in image_entries) / total, 4
    )
    normalized = np.round(total / count, 4)
    return {"num_pistils": total, "avg_green_ratio": avg_green, "avg_orange_ratio": avg_orange, "num_pistils_normalized": normalized}

def aggregate_folder_json(numbered_folder_path):
    """
    For a given numbered folder (which represents a flower),
    scan through its image subfolders (those starting with "IMG_")
    and collect their per-image aggregated JSON files from
    "pistils_analysis/aggregated_pistils_color_ratios.json".
    
    Then, compute the weighted aggregate of all images and always
    overwrite the folder-level JSON with the new results.
    
    Returns the aggregated dictionary.
    """
    image_entries = []
    for item in os.listdir(numbered_folder_path):
        image_folder_path = os.path.join(numbered_folder_path, item)
        if os.path.isdir(image_folder_path) and item.startswith("IMG_"):
            img_json_path = os.path.join(image_folder_path, "pistils_analysis", "aggregated_pistils_color_ratios.json")
            if os.path.exists(img_json_path):
                with open(img_json_path, "r") as f:
                    img_json = json.load(f)
                image_entries.append(get_color_ratios(img_json))
            else:
                print(f"Missing per-image JSON for folder: {image_folder_path}")
    
    aggregated = aggregate_folder_entries(image_entries)
    
    folder_analysis_dir = os.path.join(numbered_folder_path, "pistils_analysis")
    os.makedirs(folder_analysis_dir, exist_ok=True)
    agg_json_path = os.path.join(folder_analysis_dir, "aggregated_pistils_color_ratios.json")
    with open(agg_json_path, "w") as f:
        json.dump(aggregated, f, indent=4)
    print(f"Overwritten aggregated JSON for folder: {numbered_folder_path}")
    return aggregated

def collect_pistils_data(base_path, day_folders):
    """
    Walk through each day folder (under base_path) and its "greenhouse" and "lab" subfolders.
    For each numbered folder, compute (or load) the folder-level aggregated JSON
    (it will be overwritten with the new results).
    Also collect per-image data from each image folder.
    
    Returns two lists:
      - data_per_image: one entry per image folder.
      - data_per_folder: one entry per numbered folder.
    """
    data_per_image = []
    data_per_folder = []
    
    for day_folder in day_folders:
        day_folder_path = os.path.join(base_path, day_folder)
        print(f"Processing {day_folder_path}...")
        
        for subfolder in ["greenhouse"]:
            subfolder_path = os.path.join(day_folder_path, subfolder)
            if not os.path.exists(subfolder_path):
                print(f"Missing subfolder: {subfolder_path}")
                continue
            for numbered_folder in os.listdir(subfolder_path):
                numbered_folder_path = os.path.join(subfolder_path, numbered_folder)
                if not os.path.isdir(numbered_folder_path):
                    continue
                
                # Always aggregate folder-level JSON (overwrite previous results)
                aggregated = aggregate_folder_json(numbered_folder_path)
                
                folder_entry = {
                    "day": day_folder,
                    "location": subfolder,
                    "number": numbered_folder,
                    "num_pistils": aggregated.get("num_pistils", 0),
                    "avg_green_ratio": aggregated.get("avg_green_ratio", 0),
                    "avg_orange_ratio": aggregated.get("avg_orange_ratio", 0),
                    "num_pistils_normalized": aggregated.get("num_pistils_normalized", 0)
                }
                data_per_folder.append(folder_entry)
                
                # Process per-image JSON files in this numbered folder
                for image_folder in os.listdir(numbered_folder_path):
                    image_folder_path = os.path.join(numbered_folder_path, image_folder)
                    if os.path.isdir(image_folder_path) and image_folder.startswith("IMG_"):
                        img_json_path = os.path.join(image_folder_path, "pistils_analysis", "aggregated_pistils_color_ratios.json")
                        if os.path.exists(img_json_path):
                            with open(img_json_path, "r") as img_f:
                                img_json = json.load(img_f)
                            image_entry = {
                                "day": day_folder,
                                "location": subfolder,
                                "number": numbered_folder,
                                "image": image_folder,
                                "num_pistils": img_json.get("num_pistils", 0),
                                "avg_green_ratio": np.round(img_json.get("average_green_ratio", img_json.get("avg_green_ratio", 0)), 4),
                                "avg_orange_ratio": np.round(img_json.get("average_orange_ratio", img_json.get("avg_orange_ratio", 0)), 4)
                            }
                            data_per_image.append(image_entry)
                        else:
                            print(f"Missing per-image aggregated JSON for folder: {image_folder_path}")
    return data_per_image, data_per_folder

def save_data_to_csv(data, output_file):
    if not data:
        print(f"No data to save for {output_file}")
        return
    fieldnames = list(data[0].keys())
    with open(output_file, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in data:
            writer.writerow(row)


if __name__ == "__main__":
    # List the day folders in your experiment (modify as needed)
    exp_2_day_folders = [
        "day_1_2024_12_05",
        "day_2_2024_12_09",
        "day_3_2024_12_12",
        "day_4_2024_12_17",
        "day_5_2024_12_24",
        "day_6_2024_12_30",
        "day_7_2025_01_06",
        "day_8_2025_01_09",
        "day_9_2025_01_16",
    ]
    # Base path where pistils analysis results are stored
    base_path = "/sise/shanigu-group/etaylor/assessing_cannabis_exp/experiment_2/results/faster_rcnn_with_yolo"
    # Output folder for CSV results
    output_folder = os.path.join(base_path, "csv_results")
    os.makedirs(output_folder, exist_ok=True)

    data_per_image, data_per_folder = collect_pistils_data(base_path, exp_2_day_folders)

    save_data_to_csv(
        data_per_image,
        os.path.join(output_folder, "collected_pistils_results_per_image.csv"),
    )
    save_data_to_csv(
        data_per_folder,
        os.path.join(output_folder, "collected_pistils_results_per_folder.csv"),
    )

    print("Data collection complete. CSV files saved as:")
    print(" - collected_pistils_results_per_image.csv")
    print(" - collected_pistils_results_per_folder.csv")
