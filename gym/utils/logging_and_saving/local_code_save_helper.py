import os
import shutil
import fnmatch
from gym import LEGGED_GYM_ROOT_DIR


def configure_local_files(log_dir, save_paths):
    def create_ignored_pattern_except(*patterns):
        def _ignore_patterns(path, names):
            keep = set(
                name for pattern in patterns for name in fnmatch.filter(names, pattern)
            )
            ignore = set(
                name
                for name in names
                if name not in keep and not os.path.isdir(os.path.join(path, name))
            )
            return ignore

        return _ignore_patterns

    def remove_empty_folders(path, removeRoot=True):
        if not os.path.isdir(path):
            return
        # remove empty subfolders
        files = os.listdir(path)
        if len(files):
            for f in files:
                fullpath = os.path.join(path, f)
                if os.path.isdir(fullpath):
                    remove_empty_folders(fullpath)
        # if folder empty, delete it
        files = os.listdir(path)
        if len(files) == 0 and removeRoot:
            os.rmdir(path)

    # copy the relevant source files to the local logs for records
    save_dir = log_dir + "/files/"
    for save_path in save_paths:
        if save_path["type"] == "file":
            os.makedirs(save_dir + save_path["target_dir"], exist_ok=True)
            shutil.copy2(save_path["source_file"], save_dir + save_path["target_dir"])
        elif save_path["type"] == "dir":
            include = save_path["include_patterns"]
            shutil.copytree(
                save_path["source_dir"],
                save_dir + save_path["target_dir"],
                ignore=create_ignored_pattern_except(*include),
            )
        else:
            print("WARNING: uncaught save path type:", save_path["type"])
    remove_empty_folders(save_dir)


def save_local_files_to_logs(log_dir):
    save_paths = get_local_save_paths()
    configure_local_files(log_dir, save_paths)


def check_local_saving_flag(train_cfg):
    """Check if enable_local_saving is set to true in the training_config"""

    if hasattr(train_cfg, "logging") and hasattr(
        train_cfg.logging, "enable_local_saving"
    ):
        enable_local_saving = train_cfg.logging.enable_local_saving
    else:
        enable_local_saving = False
    return enable_local_saving


def get_local_save_paths():
    """Create a save_paths object for saving code locally"""

    learning_dir = os.path.join(LEGGED_GYM_ROOT_DIR, "learning")
    learning_target = os.path.join("learning")

    gym_dir = os.path.join(LEGGED_GYM_ROOT_DIR, "gym")
    gym_target = os.path.join("gym")

    # list of things to copy
    # source paths need the full path and target are relative to log_dir
    save_paths = [
        {
            "type": "dir",
            "source_dir": learning_dir,
            "target_dir": learning_target,
            "include_patterns": ("*.py", "*.json"),
        },
        {
            "type": "dir",
            "source_dir": gym_dir,
            "target_dir": gym_target,
            "include_patterns": ("*.py", "*.json"),
        },
    ]

    return save_paths
