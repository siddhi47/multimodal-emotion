# multimodal-emotion
Multimodal learning using IMEOCAP dataset.

# Configuration
Refer to config.ini to configure the paths of audio and text dataset. The config file can also be used to add hyperparameters such as learning rate and dropout.

# Usage
Clone the code, configure the config.ini file. Run python emotion_utils\main.py. The main.py file also takes some arguments. For more details run 

```python
python emotion_utils\main.py --help
```

# SLURM job
Use the slurm job (lightning_job.sh) as a template to run on HPC. You can also check the usage of code in the slurm job file.


The dataset is accessible here:https://sail.usc.edu/iemocap/.
