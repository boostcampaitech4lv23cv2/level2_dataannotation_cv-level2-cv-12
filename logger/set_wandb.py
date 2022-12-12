import wandb
import json
import os

def wandb_init(args):
    with open('conf.json') as f:
        config = json.load(f)
    wandb.init(project=config['wandb_project'], name=args.experiment_name, entity=config['wandb_entity'])
    wandb.config.update(args)
    os.makedirs(os.path.join(args.model_dir, args.experiment_name), exist_ok=True)