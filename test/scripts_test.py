#!/usr/bin/env python
# -*- coding: utf-8 -*-
import glob
import os
import runpy

import natsort
import pytest


PROJECT_DIR = os.path.realpath(os.path.dirname(os.path.dirname(__file__)))


def get_paths_of_scripts(path_glob, exclude_sub_strings):
    plot_script_paths = glob.glob(path_glob)
    plot_script_paths_sorted = natsort.natsorted(plot_script_paths)
    plot_script_paths_sorted_reduced = [
        p
        for p in plot_script_paths_sorted
        if not any(sub in p for sub in exclude_sub_strings)
    ]
    return plot_script_paths_sorted_reduced


class Test_scripts_in_main_dir:
    @pytest.mark.parametrize(
        "path_script",
        get_paths_of_scripts(
            path_glob=os.path.join(PROJECT_DIR, "s*.py"),
            exclude_sub_strings=[
                "setup",
            ],
        ),
    )
    def test_execute_scripts(self, path_script):

        print(f"Execut script:\n{path_script}")
        runpy.run_path(path_script, init_globals={}, run_name="__main__")


class Test_scripts_in_scripts_dir:
    @pytest.mark.parametrize(
        "path_script",
        get_paths_of_scripts(
            path_glob=os.path.join(
                PROJECT_DIR,
                "scripts",
                "s*.py",
            ),
            exclude_sub_strings=[
                "s021",
                "s022",
                "s040",
                "s101",
                "s103",
                "s104",
                "s105",
                "s107",
                "s112",
                "s137",
            ],
        ),
    )
    def test_execute_scripts(self, path_script):

        print(f"Execut script:\n{path_script}")
        runpy.run_path(path_script, init_globals={}, run_name="__main__")
