from src.classification.utils import image_classification_models


def train_model_across_dataloaders(dataloaders_dict, model_name):
    from fastai.metrics import Precision, Recall, RocAucMulti
    from fastai.vision.all import vision_learner

    # Define the model architecture and metrics
    precision_macro = Precision(average="macro")
    recall_macro = Recall(average="macro")

    results = {}
    models = {}

    for name, dls in dataloaders_dict.items():
        print(f"Training with configuration: {name}")

        model = vision_learner(
            dls=dls,
            arch=image_classification_models[model_name],
            metrics=[error_rate, precision_macro, recall_macro],
        )

        # Train the model
        model.fine_tune(epochs=25)

        # Get and store the results
        results[name] = model.validate()
        models[name] = model

    # Display the results
    for config, result in results.items():
        print(f"Results for {config}: {result}")

    return results, models
