from imagen_pytorch import Unet, SRUnet256, Imagen, ImagenTrainer
from imagen_pytorch.data import Dataset
from tqdm import tqdm


if __name__ == "__main__":
    image_folder = r"../datasets/flowers"
    results_folder = r"results/flowers/200000",
    train_num_steps = 200000

    # unets for unconditional imagen
    unet1 = Unet(
        dim = 32,
        dim_mults = (1, 2, 4, 8),
        num_resnet_blocks = 1,
        layer_attns = (False, False, False, True),
        layer_cross_attns = False
    )

    unet2 = SRUnet256(
        dim = 32,
        dim_mults = (1, 2, 4),
        num_resnet_blocks = (2, 4, 8),
        layer_attns = (False, False, True),
        layer_cross_attns = False
    )

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
        use_lion = True,
        lr = 0.0001,
        eps = 1e-8,
        beta1 = 0.9,
        beta2 = 0.99,
    ).cuda()

    # instantiate your dataloader, which returns the necessary inputs to the DDPM as tuple in the order of images, text embeddings, then text masks. in this case, only images is returned as it is unconditional training
    dataset = Dataset(
        folder = image_folder,
        image_size = 256
    )

    trainer.add_train_dataset(dataset, batch_size = 16)

    # resume from latest checkpoint
    # trainer.load_from_checkpoint_folder()

    # working training loop
    from torch.utils.tensorboard import SummaryWriter
    tb_writer = SummaryWriter(log_dir=results_folder)

    with tqdm(initial = 0, total = train_num_steps, disable = not trainer.is_main) as pbar:
        for i in range(train_num_steps):
            loss = trainer.train_step(unet_number = 1, max_batch_size = 4)
            pbar.set_description(f'loss: {loss:.4f}')
            # tensorboard log
            tb_writer.add_scalar("loss", loss, i)

            if not (i % 100):
                valid_loss = trainer.valid_step(unet_number = 1, max_batch_size = 4)
                trainer.print(f'valid loss: {valid_loss}')
                # tensorboard log
                tb_writer.add_scalar("valid loss", valid_loss, i)

            if not (i % 1000) and trainer.is_main: # is_main makes sure this can run in distributed
                # gen image
                images = trainer.sample(batch_size = 1, return_pil_images = True) # returns List[Image]
                images[0].save(f'{results_folder}/sample-{i // 100}.png')

                # save model
                trainer.save_to_checkpoint_folder()

            pbar.update(1)

    tb_writer.close()
    trainer.print("training complete!")
