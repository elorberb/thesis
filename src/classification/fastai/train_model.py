import argparse
import json
from fastai.vision.all import *
from sklearn.metrics import precision_score, recall_score, accuracy_score

TRAIN_DATASET_PATH = '/home/etaylor/code_projects/thesis/segments/etaylor_cannabis_patches_train_26-04-2024_15-44-44/trichome_dataset'
TEST_DATASET_PATH = '/home/etaylor/code_projects/thesis/segments/etaylor_cannabis_patches_test_26-04-2024_15-44-44/trichome_dataset'

# Define available models
available_models = {
    'resnet34': models.resnet34,
    'resnet50': models.resnet50,
    'resnet101': models.resnet101,
    'resnet152': models.resnet152,
    'resnext50_32x4d': models.resnext50_32x4d,
    'resnext101_32x8d': models.resnext101_32x8d,
    'wide_resnet50_2': models.wide_resnet50_2,
    'wide_resnet101_2': models.wide_resnet101_2,
    'vgg16_bn': models.vgg16_bn,
    'vgg19_bn': models.vgg19_bn,
    'squeezenet1_1': models.squeezenet1_1,
    'densenet121': models.densenet121,
    'densenet169': models.densenet169,
    'densenet201': models.densenet201,
    'densenet161': models.densenet161,
}

def train_and_evaluate(model_name, model_func, train_path, test_path, epochs, output_path):
    print(f"Training and evaluating {model_name}...")

    # Load the datasets
    dls = ImageDataLoaders.from_folder(train_path, valid_pct=0.2, item_tfms=Resize(24))
    
    # Load test dataset separately and ensure it includes labels
    test_files = get_image_files(test_path)
    test_dl = dls.test_dl(test_files, with_labels=True)

    # Create the learner
    learn = vision_learner(dls, model_func, metrics=[error_rate, Precision(average='macro'), Recall(average='macro')])

    # Fine-tune the model
    learn.fine_tune(epochs)

    # Collect training losses
    train_losses = [loss.item() for loss in learn.recorder.losses]

    # Evaluate on the test set
    preds, targs = learn.get_preds(dl=test_dl)
    
    if targs is not None:
        pred_classes = preds.argmax(dim=1).numpy()
        true_classes = targs.numpy()

        precision = precision_score(true_classes, pred_classes, average='macro')
        recall = recall_score(true_classes, pred_classes, average='macro')
        accuracy = accuracy_score(true_classes, pred_classes)
    else:
        precision = recall = accuracy = None

    results = {
        'model': model_name,
        'precision': precision,
        'recall': recall,
        'accuracy': accuracy
    }

    # Print results to the terminal
    print(json.dumps(results, indent=4))

    # Save results to JSON file
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_name', type=str, help='Name of the model')
    parser.add_argument('epochs', type=int, help='Number of epochs to train')
    args = parser.parse_args()

    if args.model_name in available_models:
        model_func = available_models[args.model_name]
    else:
        raise ValueError(f"Model name must be one of {list(available_models.keys())}")
    
    output_path = f"/home/etaylor/code_projects/thesis/src/classification/fastai/models_scores/{args.model_name}_results.json"
    
    train_and_evaluate(args.model_name, model_func, TRAIN_DATASET_PATH, TEST_DATASET_PATH, args.epochs, output_path)
