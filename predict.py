import torch
from imagen_pytorch import Unet, BaseUnet64, SRUnet256, SRUnet1024, Imagen, ImagenTrainer


if __name__ == "__main__":
    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )

    # unets for unconditional imagen
    unet1 = BaseUnet64(
        dim = 64,
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

    # unet3 = SRUnet1024(
    #     dim = 32,
    #     dim_mults = (1, 2, 4),
    #     num_resnet_blocks = (2, 4, 8),
    #     layer_attns = (False, False, True),
    #     layer_cross_attns = False
    # )

    unets = (unet1, unet2)

    # imagen, which contains the unet above
    imagen = Imagen(
        condition_on_text = False,      # this must be set to False for unconditional Imagen
        unets = unets,
        image_sizes = (64, 256),
        timesteps = 1000
    ).to(device)

    # load weight
    runs = "800000"
    state_dict: dict = torch.load(f"results/flowers/checkpoint.{runs}.pt")
    print(state_dict.keys())  # ['step', 'model', 'opt', 'ema', 'scaler', 'version']

    # 是否使用ema model
    use_ema = True

    if not use_ema:
        imagen.load_state_dict(state_dict["model"])
    else:
        unet_number = len(unets)
        ema_dict: dict = state_dict["ema"]

        # convert ema model keys
        new_ema_dict = {}
        for number in range(unet_number):
            for ema_key, ema_value in ema_dict.items():
                ema_key_prefix = f"{number}.ema_model"
                if ema_key_prefix in ema_key:
                    dst_key = ema_key.replace(ema_key_prefix, f"unets.{number}")
                    new_ema_dict[dst_key] = ema_value
        imagen.load_state_dict(new_ema_dict)

    # sample image
    with torch.inference_mode():
        images = imagen.sample(batch_size=25, return_pil_images=True)

    for i, image in enumerate(images):
        image.save(f"{i}.png")
