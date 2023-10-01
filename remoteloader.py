import os
import json
import hashlib
import urllib
from urllib import request
import folder_paths
from custom_nodes.DTAIComfyLoaders import config


class RemoteLoader:
    def __init__(self, path, uri):
        self.data = {}
        self.uri = uri
        self.path = path

        self.load()

    def load(self):
        try:
            with urllib.request.urlopen(self.uri) as f:
                self.data = json.loads(f.read().decode('utf-8'))
            print(f"Loaded {self.path} from {self.uri}")
        except Exception as e:
            print(f"Failed to load {self.uri} for {self.path}: {e}")

    def list(self):
        try:
            return list(self.data.keys())
        except AttributeError:
            return []

    def filename(self, key):
        # if download_path is set, use it joined with self.path
        if '' != config.download_path:
            folder = os.path.join(config.download_path, self.path)
        else:
            # otherwise, use the default path
            folder = folder_paths.get_folder_paths(self.path)[0]

        full_path = os.path.join(folder, key)
        # get absolute path for the full path
        full_path = os.path.abspath(full_path)
        print("full_path: ", full_path)
        # if the file exists return full_path
        if os.path.exists(full_path):
            return full_path

        if key not in self.data:
            raise KeyError(f"Key {key} not found in {self.uri}")

        key = self.data[key]

        filename = ""
        # if key is a url with a filename with any extension at the end use it
        # verify this by checking if the key is a url and the last segment includes a .
        if key.startswith("http") and key.split("/")[-1].find(".") > 0:
            filename = key.split("/")[-1]
            # strip off any query params from filename
            filename = filename.split("?")[0]
        else:
            # Use hashlib to create the md5 hash of the key
            hash_object = hashlib.md5(key.encode())
            md5_hash = hash_object.hexdigest()
            filename = md5_hash

        # if download_path is set, use it joined with self.path
        if '' != config.download_path:
            folder = os.path.join(config.download_path, self.path)
        else:
            # otherwise, use the default path
            folder = folder_paths.get_folder_paths(self.path)[0]

        full_path = os.path.join(folder, filename)

        # Combine self.path with md5_hash to get the filepath
        return full_path

    def download(self, key):
        filename = self.filename(key)

        # If the file doesn't exist at that path, download and save it
        if not os.path.exists(filename):
            if key not in self.data:
                raise KeyError(f"Key {key} not found in {self.uri}")
            urllib.request.urlretrieve(self.data[key], filename)

        return filename