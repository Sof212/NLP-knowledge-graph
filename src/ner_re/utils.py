import json
import os
import random

import numpy as np
import yaml


def set_seed(seed):
    """
    Seed Python, NumPy and (if installed) PyTorch random generators.

    Parameters
    ----------
    seed : int
        Seed value.

    Returns
    -------
    None
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_device():
    """
    Return the best available PyTorch device.

    Returns
    -------
    str
        ``"cuda"`` if a GPU is available, otherwise ``"cpu"``.
    """
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def load_config(path):
    """
    Load a YAML configuration file.

    Parameters
    ----------
    path : str
        Path to the YAML file.

    Returns
    -------
    dict
        Parsed configuration.
    """
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path):
    """
    Create a directory (and its parents) if it does not exist.

    Parameters
    ----------
    path : str
        Directory to create.

    Returns
    -------
    str
        The same path.
    """
    os.makedirs(path, exist_ok=True)
    return path


def save_json(obj, path):
    """
    Save a JSON-serialisable object to disk.

    Parameters
    ----------
    obj : object
        Object to save.
    path : str
        Output file path.

    Returns
    -------
    None
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path):
    """
    Load a JSON file.

    Parameters
    ----------
    path : str
        Input file path.

    Returns
    -------
    object
        Parsed content.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
