import torch
import argparse
from diffusers.utils import load_image, check_min_version
from controlnet_flux import FluxControlNetModel
from transformer_flux import FluxTransformer2DModel
from pipeline_flux_controlnet_inpaint import FluxControlNetInpaintingPipeline
import os

check_min_version("0.30.2")

def main(image_path, mask_path, prompt, save_path):
    # Build pipeline
    controlnet = FluxControlNetModel.from_pretrained("alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Alpha", torch_dtype=torch.bfloat16)
    transformer = FluxTransformer2DModel.from_pretrained(
            "black-forest-labs/FLUX.1-dev", subfolder='transformer', torch_dtype=torch.bfloat16
        )
    pipe = FluxControlNetInpaintingPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        controlnet=controlnet,
        transformer=transformer,
        torch_dtype=torch.bfloat16
    ).to("cuda")
    pipe.transformer.to(torch.bfloat16)
    pipe.controlnet.to(torch.bfloat16)

    # Load image and mask
    size = (768, 768)
    image = load_image(image_path).convert("RGB").resize(size)
    mask = load_image(mask_path).convert("RGB").resize(size)
    generator = torch.Generator(device="cuda").manual_seed(24)

    # Inpaint
    result = pipe(
        prompt=prompt,
        height=size[1],
        width=size[0],
        control_image=image,
        control_mask=mask,
        num_inference_steps=28,
        generator=generator,
        controlnet_conditioning_scale=0.9,
        guidance_scale=3.5,
        negative_prompt="",
        true_guidance_scale=1.0 # default: 3.5 for alpha and 1.0 for beta
    ).images[0]

    # Make save_path directory if it doesn't exist
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    result.save(save_path)
    print("Successfully inpainted image")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flux ControlNet Inpainting")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image")
    parser.add_argument("--mask_path", type=str, required=True, help="Path to the mask image")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt for inpainting")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save the output image")

    args = parser.parse_args()
    main(args.image_path, args.mask_path, args.prompt, args.save_path)


# Run like this: python main_naren.py --image_path <image_path> --mask_path <mask_path> --prompt <prompt> --save_path <save_path>
