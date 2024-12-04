#!/bin/bash

# Define the image name
image_name="road2"

# Set the paths and prompt
source_image="source_images/road/${image_name}.jpg"
mask_image="mask_images/${image_name}_mask.png"
text_prompt="brown shaded left side of the road"
save_path="output_images/${image_name}_output.png"

# Run the Python script with the specified arguments
python main_naren.py --image_path "$source_image" --mask_path "$mask_image" --prompt "$text_prompt" --save_path "$save_path"