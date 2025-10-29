import os
import shutil


def clean_pycache(path):
    for root, dirs, files in os.walk(path):
        if '__pycache__' in dirs:
            shutil.rmtree(os.path.join(root, '__pycache__'))
            print(f"Deleted: {os.path.join(root, '__pycache__')}")
        for file in files:
            if file.endswith('.pyc'):
                os.remove(os.path.join(root, file))
                print(f"Deleted: {os.path.join(root, file)}")


if __name__ == "__main__":
    clean_pycache(os.getcwd())