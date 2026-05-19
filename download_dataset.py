import os
import shutil
import kagglehub

path = kagglehub.dataset_download("alessiocorrado99/animals10")
print(f"Downloaded to: {path}")

# Move all contents to current directory
for item in os.listdir(path):
    src = os.path.join(path, item)
    dst = os.path.join(os.getcwd(), item)
    if os.path.isdir(src):
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dst)

print("Copied to current directory")