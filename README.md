# Train RL agents to play Pokemon Emerald

## Training the Model 🏋️ 

1. Install dependencies:  
```pip install -r baselines/requirements.txt```

2. Run training script:  
```python baselines/run_baseline_parallel_fast.py```

## Tracking Training Progress 📈 
The current state of each game is rendered to images in the session directory.   
You can track the progress in tensorboard by moving into the session directory and running:  
```tensorboard --logdir .```  
You can then navigate to `localhost:6006` in your browser to view metrics.  
To enable wandb integration, change `use_wandb_logging` in the training script to `True`.
