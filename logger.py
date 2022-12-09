import wandb

def init_wandb(config : dict):
    wandb.init(project="Experiments", entity="cv12-data-production", name=config.experiment_name, config=config, reinit=True)
    
    # wandb_config = {
    #     "epochs": config.max_epoch,
    #     "batch_size": config.batch_size,
    #     "learning_rate": config.learning_rate,
    #     "image_size": config.image_size,
    #     "input_size": config.input_size,
    #     "optimizer": config.optimizer,
    #     "scheduler": config.scheduler,
    #     }

    # wandb.run.name = config.experiment_name
    # wandb.run.save()

    wandb.define_metric("learning_rate")
    wandb.define_metric("cls_loss", summary="min")
    wandb.define_metric("angle_loss", summary="min")
    wandb.define_metric("iou_loss", summary="min")
    wandb.define_metric("mean_loss", summary="min")
    print("wanbd init done.")

def logging(learning_rate, cls_loss, angle_loss, iou_loss, mean_loss):
    wandb.log({
    "learning_rate": learning_rate,
    "cls_loss": cls_loss,
    "angle_loss": angle_loss,
    "iou_loss": iou_loss,
    "mean_loss": mean_loss
    })

def finish():
    wandb.finish()