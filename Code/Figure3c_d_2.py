from PIL import Image
import numpy as np
from scipy.ndimage import map_coordinates
import os

def transform_to_parallelogram(image_path, output_path, skew_factor=0.25):

    try:
        img = Image.open(image_path)
        img_array = np.array(img)
        height, width = img_array.shape[:2]
        
        extra_width = int(height * skew_factor) + 10  
        
        new_width = width + extra_width
        if len(img_array.shape) > 2:
            output = np.zeros((height, new_width, img_array.shape[2]), dtype=img_array.dtype)
        else:
            output = np.zeros((height, new_width), dtype=img_array.dtype)

        y_out, x_out = np.mgrid[0:height, 0:new_width]
        

        x_in = x_out - skew_factor * (height - y_out)
        y_in = y_out
        

        valid = (x_in >= 0) & (x_in < width) & (y_in >= 0) & (y_in < height)
        

        if len(img_array.shape) > 2:
            for i in range(img_array.shape[2]):
                output[y_out[valid], x_out[valid], i] = map_coordinates(img_array[:, :, i], [y_in[valid], x_in[valid]], order=1)
        else:
            output[y_out[valid], x_out[valid]] = map_coordinates(img_array, [y_in[valid], x_in[valid]], order=1)
        

        output_img = Image.fromarray(output)
        

        output_img.save(output_path)

        return True
    except Exception as e:
        return False

def batch_process_images(base_dir, types, regions, skew_factor=0.25):

    successful = 0
    failed = 0
    
    for diff_type in types:
        for region in regions:
            input_path = os.path.join(base_dir, f"diff_{diff_type}_{region}_precipitation.png")
            output_path = os.path.join(base_dir, f"diff_{diff_type}_{region}_parallelogram.png")

            if not os.path.exists(input_path):
                failed += 1
                continue

            if transform_to_parallelogram(input_path, output_path, skew_factor):
                successful += 1
            else:
                failed += 1
    

if __name__ == "__main__":
    base_directory = "./"
    types_to_process = ['weak', 'strong']
    regions_to_process = ['EP', 'NA', 'WP', 'NI', 'SI', 'SP']

    batch_process_images(base_directory, types_to_process, regions_to_process, skew_factor=0.45)