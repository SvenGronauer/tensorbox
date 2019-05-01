import os


def mkdir(path):
    created_dir = False
    if not os.path.exists(path):
        os.mkdir(path)
        created_dir = True
    return created_dir