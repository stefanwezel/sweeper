import os
import logging
from zipfile import ZipFile
from typing import List, Tuple
import rawpy
from PIL import Image


class FileClient:
    """Class to handle file operations such as creating, removing and zipping directories."""

    def __init__(
        self,
        media_folder: str,
        sweep_session_id: str,
    ) -> None:
        self.media_folder = media_folder
        self.sweep_session_id = sweep_session_id
        self.upload_dir = os.path.join(self.media_folder, self.sweep_session_id)

    def create_dir(self) -> None:
        """Create new dir in media_folder with name sweep_session_id."""
        new_dir: str = self.upload_dir
        assert not os.path.exists(new_dir)
        os.mkdir(new_dir)

    def remove_directory(self) -> None:
        dir_to_remove: str = self.upload_dir
        zip_to_remove: str = os.path.join(
            self.media_folder, f"{self.sweep_session_id}.zip"
        )
        assert os.path.exists(dir_to_remove)

        try:
            os.remove(zip_to_remove)
            logging.info(f"Zipfile '{zip_to_remove}' successfully removed.")
        except FileNotFoundError as e:
            logging.info(f"No file {zip_to_remove} found: {e.strerror}")
        try:
            # Iterate over all files and subdirectories in the directory
            for root, dirs, files in os.walk(dir_to_remove, topdown=False):
                for file in files:
                    file_path = os.path.join(root, file)
                    os.remove(file_path)  # Remove each file

                for dir in dirs:
                    dir_path = os.path.join(root, dir)
                    os.rmdir(dir_path)  # Remove each subdirectory
            # After all files and subdirectories are removed, remove the empty directory itself
            os.rmdir(dir_to_remove)
            logging.info(f"Directory '{dir_to_remove}' successfully removed.")
        except OSError as e:
            logging.info(f"Error: {dir_to_remove} : {e.strerror}")

    def zip_dir(self, subset: List[str]) -> str:
        zip_filename: str = f"{self.sweep_session_id}.zip"
        zip_filepath: str = os.path.join(self.media_folder, zip_filename)
        with ZipFile(zip_filepath, "w") as zip:
            for file in subset:
                zip.write(file, os.path.basename(file))

        return zip_filename


def convert_dng_to_jpg(dng_path: str) -> Tuple[str, str]:
    # Open the DNG file
    with rawpy.imread(dng_path) as raw:
        # Convert to RGB array
        rgb = raw.postprocess()

    # Create a PIL Image object from the RGB array
    img = Image.fromarray(rgb)
    # Get the directory and filename of the DNG file
    directory, filename = os.path.split(dng_path)
    # Generate the path for the JPG file in the same directory
    jpg_path = os.path.join(directory, os.path.splitext(filename)[0] + ".jpg")
    # Save the PIL Image as a JPG file
    img.save(jpg_path)

    return jpg_path, dng_path


def strip_media_folder_from_path(media_folder: str, path: str) -> str:
    return path.replace(media_folder + "/", "")
