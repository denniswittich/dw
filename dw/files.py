"""
Created on Oct 10 2018
Last edited on Oct 10 2018
@author: Dennis Wittich
"""

import os


def crawl(folder, action_condition, action, skip_condition=None):
    file_list = os.listdir(folder)
    for file in file_list:
        joined = os.path.join(folder, file)
        if skip_condition and skip_condition(joined):  # ---------------------- skip file/folder if skip condition
            print('skip!')
            continue
        elif os.path.isdir(joined):  # ---------------------------------------- step into subdir
            crawl(joined, action_condition, action, skip_condition)
        elif action_condition(joined):  # ------------------------------------- perform action if action condition
            print('action!')
            action(joined)


def ensure_folder_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


def rename_endings(folder, old_ending, new_ending):
    chars = len(old_ending)
    for file in os.listdir(folder):
        if file.endswith(old_ending):
            os.rename(file, file[:-chars] + new_ending)
