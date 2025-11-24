import os
import shutil
from PIL import Image

def delete_folder_contents(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

def split_image(input_image_path, tile_size=(50, 50)):


    image = Image.open(input_image_path)
    image = image.resize((1024,1024))
    width, height = image.size
    num_tiles_x = width // tile_size[0]
    num_tiles_y = height // tile_size[1]
    tile_count = 0
    all_image = []
    for i in range(num_tiles_y):
        for j in range(num_tiles_x):
            left = j * tile_size[0]
            upper = i * tile_size[1]
            right = left + tile_size[0]
            lower = upper + tile_size[1]
            tile = image.crop((left, upper, right, lower))
           # tile.save(os.path.join(output_folder, f"{tile_count}.png"))
            tile_count += 1 
            all_image.append(tile)
    return all_image



def combine_tiles(all_image, output_image_path, tile_size):

    #tile_files = sorted(os.listdir(input_folder), key=lambda x: int(os.path.splitext(x)[0]))

    
    num_tiles_x = int((1024 // tile_size[0]))
    num_tiles_y = int((1024 // tile_size[1]))
    new_image = Image.new("RGB", (num_tiles_x * tile_size[0], num_tiles_y * tile_size[1]))

    tile_count = 0
    for i in range(num_tiles_y):
        for j in range(num_tiles_x):
            
            tile = all_image[tile_count]
            new_image.paste(tile, (j * tile_size[0], i * tile_size[1]))
            tile_count += 1  

    #new_image.save(output_image_path)

    return new_image

