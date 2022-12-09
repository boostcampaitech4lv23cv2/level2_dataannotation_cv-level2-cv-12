import wandb
import os

def wandb_init(args):
    wandb.init(project="seok", name=args.experiment_name, entity="cv12-data-production")
    wandb.config.update(args)
    os.makedirs("trained_models/"+args.experiment_name, exist_ok=True)