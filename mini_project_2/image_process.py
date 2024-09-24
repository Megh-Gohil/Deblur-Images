from PIL import Image
import os

def downscale_images(folder_path):
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.png'):
                file_path = os.path.join(root, file)
                try:
                    img = Image.open(file_path)
                    # Downscaling images to 448, 256. img.resize uses dimension as (W, H).
                    img = img.resize((448, 256), Image.LANCZOS)
                    img.save(file_path)
                    print(f"Downscaled and saved: {file_path}")
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

if __name__ == "__main__":
    train_sharp_path = os.path.join(os.path.dirname(__file__), "train", "train_sharp")
    downscale_images(train_sharp_path)


