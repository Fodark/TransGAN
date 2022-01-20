import omegaconf
from train_derived import main
import wandb


if __name__ == '__main__':
    config = omegaconf.OmegaConf.load('./configs/celeba_256.yaml')
    # init wandb
    wandb.init(project="TransGAN", name="celeba_256", config=config)
    main(config)