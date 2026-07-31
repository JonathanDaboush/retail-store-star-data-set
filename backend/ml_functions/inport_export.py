# ============================================================
# PICKLE SAVE / LOAD HELPERS
# ============================================================

def save_pickle(obj, path):
    """Save a Python object to a pickle file."""

    folder = os.path.dirname(path)

    if folder:
        os.makedirs(folder, exist_ok=True)

    with open(path, "wb") as file:
        pickle.dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved artifact: {path}")


def load_pickle(path):
    """Load a Python object from a pickle file."""

    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing file: {path}")

    with open(path, "rb") as file:
        obj = pickle.load(file)

    print(f"Loaded artifact: {path}")

    return obj


# ============================================================
# MODEL EXPORT / IMPORT WRAPPERS
# ============================================================

def export_model_package(package, folder, filename):

    os.makedirs(folder, exist_ok=True)

    if not filename.endswith(".pkl"):
        filename += ".pkl"

    path = os.path.join(folder, filename)

    save_pickle(package, path)

    print(f"Exported model package: {path}")

    return path


def import_model_package(folder, filename):

    path = os.path.join(folder, filename)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Model package not found: {path}")

    package = load_pickle(path)

    print(f"Loaded model package: {path}")

    return package
