"""
Created on Oct 10 2018
Last edited on Oct 10 2018
@author: Dennis Wittich
"""

import os


def crawl_do(root, condition, action, skip=None, verbose=False):
    """Recursively crawls folders and performs action on files which fulfil a condition

        Parameters
        ----------
        root : str
            Path to root of folder tree to crawl
        condition : function(file_path)
            Function which checks whether action should be used on file
        action : function(file_path)
            Function(file_path) which should be applied to files
        skip : function(file_path):bool | None
            Function which

        Returns
        -------
        out: None

        Notes
        -----
        'I' will always have 3 dimensions: (rows, columns dimensions).
        Last dimension will be of length 1 or 3, depending on the image.

    """

    file_list = os.listdir(root)
    for file in file_list:
        joined = os.path.join(root, file)
        if skip and skip(joined):  # -------------------------------------- skip file/folder
            if verbose: print('skipping file', joined)
            continue
        elif os.path.isdir(joined):  # ------------------------------------ step into folder
            if verbose: print('stepping into', joined)
            crawl_do(joined, condition, action, skip, verbose)
        elif condition(joined):  # ---------------------------------------- perform action
            if verbose: print('performing action on file', joined)
            action(joined)


def ensure_folder_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


def rename_endings(folder, old_ending, new_ending):
    chars = len(old_ending)
    for file in os.listdir(folder):
        if file.endswith(old_ending):
            os.rename(file, file[:-chars] + new_ending)
