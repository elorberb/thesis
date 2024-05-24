import subprocess
import os
from src.segmentation.framework_handlers import detectron2_handler


def create_and_submit_slurm_job(model_config):
    # Set environment variables for template substitution
    os.environ['JOB_NAME'] = f"train_{model_config.replace('/', '_')}"
    os.environ['OUTPUT_DIR'] = "/home/etaylor/code_projects/thesis/src/segmentation/notebooks/detectron2/train_logs"
    os.environ['MODEL_CONFIG'] = model_config

    # Define the path to the SLURM template
    template_path = '/home/etaylor/cluster_instructions/launch_scripts/sbatch_gpu_train_detectron2_models.sh'

    # Generate the final job script by substituting the environment variables
    command = f'envsubst < {template_path} > /tmp/{os.environ["JOB_NAME"]}.sh'
    subprocess.run(command, shell=True, check=True)

    # Submit the job script to SLURM
    subprocess.run(f'sbatch /tmp/{os.environ["JOB_NAME"]}.sh', shell=True, check=True)



if __name__ == '__main__':
    
    print("Starting jobs for detectron2 detection models...")
    for model in detectron2_handler.detectron2_detection_models:
        print(f"Starting job for model: {model}")
        create_and_submit_slurm_job(model)
        
    print("Starting jobs for detectron2 segmentation models...")
    # for model in detectron2_handler.detectron2_segmentation_models:
    #     print(f"Starting job for model: {model}")
    #     create_and_submit_slurm_job(model)
