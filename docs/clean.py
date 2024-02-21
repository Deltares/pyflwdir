import os
import shutil
from pathlib import Path


def remove_dir_content(path: str) -> None:
    for root, dirs, files in os.walk(path):
        for f in files:
            os.unlink(os.path.join(root, f))
        for d in dirs:
            shutil.rmtree(os.path.join(root, d))
    if os.path.isdir(path):
        shutil.rmtree(path)


if __name__ == "__main__":
    here = os.path.dirname(__file__)
    remove_dir_content(Path(here, "_build"))
    remove_dir_content(Path(here, "_generated"))
    remove_dir_content(Path(here, "_examples"))
