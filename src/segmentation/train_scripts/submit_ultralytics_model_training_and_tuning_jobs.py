import subprocess
import os
from src.segmentation.framework_handlers import ultralytics_handler


def create_and_submit_slurm_job(model_config, train_model=True):
    # Set environment variables for template substitution
    
    os.environ['OUTPUT_DIR'] = "/home/etaylor/code_projects/thesis/src/segmentation/notebooks/ultralytics/train_logs"
    os.environ['MODEL_CONFIG'] = model_config

    if train_model:
        # Define the path to the SLURM template
        template_path = '/home/etaylor/cluster_instructions/launch_scripts/sbatch_gpu_train_ultralytics.sh'
        os.environ['JOB_NAME'] = f"train_{model_config.replace('.', '_')}"
    else: 
        # Define the path to the SLURM template
        template_path = '/home/etaylor/cluster_instructions/launch_scripts/sbatch_gpu_tune_ultralytics.sh'
        os.environ['JOB_NAME'] = f"tune_{model_config.replace('.', '_')}"
    # Generate the final job script by substituting the environment variables
    command = f'envsubst < {template_path} > /tmp/{os.environ["JOB_NAME"]}.sh'
    subprocess.run(command, shell=True, check=True)

    # Submit the job script to SLURM
    subprocess.run(f'sbatch /tmp/{os.environ["JOB_NAME"]}.sh', shell=True, check=True)



if __name__ == '__main__':
    
    print("Starting jobs for ultralytics models...")
    for model in ultralytics_handler.models:
        print(f"Starting job for model: {model}")
        create_and_submit_slurm_job(model, train_model=False)
        
