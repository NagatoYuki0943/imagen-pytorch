from imagen_pytorch import Unet, BaseUnet64, SRUnet256, SRUnet1024, Imagen, ImagenTrainer
from imagen_pytorch.data import Dataset
from tqdm import tqdm


if __name__ == "__main__":
    image_folder = r"../datasets/flowers"
    results_folder = r"results/flowers/200000"
    train_num_steps = 100000
    max_batch_size = 16

    # unets for unconditional imagen
    unet1 = BaseUnet64(
        dim = 64,
        dim_mults = (1, 2, 4, 8),
        num_resnet_blocks = 1,
        layer_attns = (False, False, False, True),
        layer_cross_attns = False
    )

    unet2 = SRUnet256(
        dim = 64,
        dim_mults = (1, 2, 4),
        num_resnet_blocks = (2, 4, 8),
        layer_attns = (False, False, True),
        layer_cross_attns = False
    )

    # unet3 = SRUnet1024(
    #     dim = 64,
    #     dim_mults = (1, 2, 4),
    #     num_resnet_blocks = (2, 4, 8),
    #     layer_attns = (False, False, True),
    #     layer_cross_attns = False
    # )

    # imagen, which contains the unet above
    imagen = Imagen(
        condition_on_text = False,      # this must be set to False for unconditional Imagen
        unets = (unet1, unet2),
        image_sizes = (64, 256),
        timesteps = 1000
    )

    trainer = ImagenTrainer(
        imagen = imagen,
        split_valid_from_train = True,  # whether to split the validation dataset from the training
        checkpoint_path = results_folder,
        checkpoint_every = 1000,
        use_lion = True,
        lr = 0.0001,
        eps = 1e-8,
        beta1 = 0.9,
        beta2 = 0.99,
    ).cuda()

    # instantiate your dataloader, which returns the necessary inputs to the DDPM as tuple in the order of images, text embeddings, then text masks. in this case, only images is returned as it is unconditional training
    dataset = Dataset(
        folder = image_folder,
        image_size = 512
    )

    trainer.add_train_dataset(dataset, batch_size = 16)

    # resume from latest checkpoint
    # trainer.load_from_checkpoint_folder()

    # working training loop
    from torch.utils.tensorboard import SummaryWriter
    tb_writer = SummaryWriter(log_dir=results_folder)

    # Train an unet at a time
    # https://github.com/lucidrains/imagen-pytorch/issues/142
    # you cannot only train on one unet at a time. you will need to save the trainer into a checkpoint, and resume training on a new unet
    unet_number = 1

    with tqdm(initial = 0, total = train_num_steps, disable = not trainer.is_main) as pbar:
        for i in range(train_num_steps):
            loss = trainer.train_step(unet_number = unet_number, max_batch_size = max_batch_size)
            trainer.update(unet_number=unet_number)

            pbar.set_description(f'loss: {loss:.4f}')
            tb_writer.add_scalar("loss", loss, i)

            if (i != 0) and (not (i % 1000)):
                valid_loss = trainer.valid_step(unet_number = unet_number, max_batch_size = max_batch_size)
                trainer.print(f'valid loss1: {valid_loss}, valid loss2: {valid_loss}')
                tb_writer.add_scalar("valid loss", valid_loss, i)

            if  (i != 0) and (not (i % 1000)) and trainer.is_main: # is_main makes sure this can run in distributed
                # gen image
                images = trainer.sample(batch_size = 1, return_pil_images = True) # returns List[Image]
                images[0].save(f'{results_folder}/sample-{i // 100}.png')

            pbar.update(1)

    tb_writer.close()
    trainer.print("training complete!")
