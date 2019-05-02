import os


def mkdir(path):
    created_dir = False
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        # os.mkdir(path)
        created_dir = True
    return created_dir