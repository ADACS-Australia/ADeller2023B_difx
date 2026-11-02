#!/usr/bin/env python3

# ----------------------------------------------------------------------------------------------------------
# Authon: Solomon Clark
# Version: 0.3.0
# Filename: config.py
# ----------------------------------------------------------------------------------------------------------
# Description: 
#
# This is a program meant to provide more efficiency and configuration capability in the experiment setup and 
# queuing process within DiFXtest_vdifsim. Meant to be run within each available test directory in order for the 
# metadata pertaining to each directory can be stored properly. Directory information can be stored within the terminal
# arguments when the file is being called. This is a bootstrap utility that will also infer CPU/GPU information
# for each test set based on the experiment directory naming conventions.
#
# The base-level functionality of this file should be to set the manifest based on the config values present
# in each test directory. Otherwise, running a -r or --reset-ini will reset all attributes present in those
# config files. 
# 
# This file should always be run with the -e --exp-root flag to indicate where the experiment directories are
# located. 
#
# To make sure the tests are synched to the manifest for execution, we will run this file at the start of 
# each run of DiFXtest.py every time that a change is made to one or more of the experiment config files.
#
# ----------------------------------------------------------------------------------------------------------

# TODO Change from .ini files to .json files
import json
import os
import subprocess
import argparse
import logging
import glob
import configparser
from dataclasses import dataclass
from pathlib import Path
import re

__title__ = "config.py"
__version__ = "0.4.0"

LOGGER = logging.getLogger(__name__)

def setup_logger(debug_out, verbose):

    # Logging-Level Config
    if debug_out:
        log_level = logging.DEBUG
    elif verbose: 
        log_level = logging.INFO
    else: 
        log_level = logging.WARNING


    logging.basicConfig(
    level=log_level, 
    format="%(levelname)s in %(filename)s: %(message)s"
    )

    LOGGER.debug("Starting logger setup...\n")

def get_testdir(): 
    return os.path.realpath(os.path.dirname(__file__))

def get_exp_path(root: str) -> Path:
    return Path(get_testdir()) / root

def count_stations(testdir: Path) -> list: 

    for vexfile in testdir.glob("*.vex"):
        
        vex_contents = vexfile.read_text(encoding="utf-8")
        vex_no_comments = re.sub(
            r"(?m)^\s*\*.*$",
            "",
            vex_contents
        )

        station_sections = re.search(
            r"(?is)\$STATION\s*;(.*?)(?=\$\w+\s*;|\Z)",
            vex_no_comments
        )

        if station_sections is not None:
            station_ids = re.findall(
                r"(?im)^\s*def\s+([^;\s]+)\s*;",
                station_sections.group(1)
            )
        
    return station_ids

def build_exp_ini(root: str):  
    # Building .ini files for each individual experiment...
    exproot = get_exp_path(root)

    # Setting default config values in each of the exp directories...
    for dir in exproot.iterdir():

        testname = os.path.basename(dir)

        LOGGER.debug(f"Printing dir names in build_exp_ini: {str(dir)}")
        ini_file = str(dir) + "/" + testname + ".ini"

        if testname.endswith(
            (".tmp", "tests.tgz", "testdata", "difx_version_info.txt")
        ):
            continue # Skip the following process if were are not somewhere that needs a .ini file...

        stations = count_stations(dir)

        config = configparser.ConfigParser()
        # .ini Configuration Attributes
        config['experiment'] = {}
        config['stations'] = {}
        config['run'] = {}
        config['comparison'] = {}
        ants = config['stations']
        comp = config['comparison']
        exp = config['experiment'] 
        exp['id'] = testname
        if testname.endswith("-gpu"): 
            exp['compute_method'] = 'gpu'
            exp['comparison_group'] = testname[:-4]
            comp['comparable_experiment'] = testname[:-4]
        else: 
            exp['compute_method'] = 'cpu'
            exp['comparison_group'] = testname
            comp['comparable_experiment'] = testname + "-gpu"

        ants['ids'] = ", ".join(stations)
        ants['num_stations'] = str(len(stations))

        run = config['run']
        run['execute'] = "true"

        # TODO Can be replaced with json.dump(data, file)
        try:
            with open(ini_file, 'w') as configfile:
                config.write(configfile)
            LOGGER.info(f"{os.path.basename(ini_file)} written successfully")
        except OSError as e: 
            LOGGER.error(f"Failed to write configuration file {configfile}:\n{e}")

        # Removing any extra .ini files from the directory on manifest reset
        old_ini_files = [
            str(inifiles)
            for inifiles in sorted(dir.glob("*.ini"))
            if os.path.basename(inifiles) != os.path.basename(ini_file)
        ]

        LOGGER.debug(f"Extra/Old .ini files detected: {old_ini_files}")

        for extrafiles in old_ini_files: 
            try: 
                LOGGER.debug(f"Removing extra file(s) {str(extrafiles)}")
                rm_command = "rm -r " + extrafiles
                proc = subprocess.run(rm_command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
                LOGGER.debug(f"Rm -r command outcode: {proc.returncode}")
            except subprocess.CalledProcessError as e:
                LOGGER.error(f"This broke for some reason: {e}")
    
def write_manifest(config: configparser): 

    manifest_file = get_testdir() + "/manifest.ini"
    manifest_config = config
    
    try:
        with open(manifest_file, 'w') as manifest: 
            manifest_config.write(manifest)
        LOGGER.info(f"{os.path.basename(manifest_file)} written successfully\n")
    except OSError as e: 
        LOGGER.error(f"Failed to write manifest config file {manifest_file}:\n{e}")

def build_manifest(root: str):

    exproot = get_exp_path(root)

    manifest_config = configparser.ConfigParser()
    manifest_config['running_experiments'] = {}

    for inifile in sorted(exproot.rglob("*.ini")):
        ini_attr = configparser.ConfigParser()
        ini_attr.read(inifile)

        if not ini_attr:
            LOGGER.warning(f"Could not read {inifile}")
            continue

        try:
            exp_name = ini_attr['experiment']['id']
            exp_run = ini_attr.getboolean('run', 'execute')
        except ValueError as ve:
            raise ValueError(
                f"Invalid Boolean value for execute attribute in {inifile}"
            ) from ve
        manifest_config['running_experiments'][exp_name] = str(exp_run).lower()

    # Have this method return the manifest cofig object to be used in other methods...
    return manifest_config

def print_manifest(manifest_config: configparser):
    # Print in-memory manifest without writing it to disk...

    if not manifest_config.has_section('running_experiments'):
        LOGGER.critical(f"Manifest config file has blank sections, aborting...")
        exit(1)
    for section in manifest_config.sections():
        print(f"[{section}]")

        for experiment, run in manifest_config[section].items():
            print(f"{experiment} = {run}")

        print()

def dry_run(root: str):

    manifest_config = build_manifest(root)
    print()
    print(f"DRY RUN: showing experiment execution manifest\n")
    print_manifest(manifest_config)

def remove_from_exec(root: str, remove_dirs: list, manifest: str):

    # run value assigned as: manifest_config['running_experiments'][exp_name] = str(exp_run).lower()

    exp_root = get_exp_path(root)

    manifest_path = Path(get_testdir()) / manifest
    manifest_config = configparser.ConfigParser()
    manifest_config.read(manifest_path)

    # Filtering for nonexistent directories
    if bad_dirs := [
            dirs
            for dirs in remove_dirs
            if not Path.is_dir(exp_root / dirs)
        ]:
            for dir in bad_dirs:
                LOGGER.error(f"\n\nDirectory '{dir}' doesn't exist in experiment root '{root}', aborting...\n")
            quit()

    if not manifest_config.has_section('running_experiments'):
        LOGGER.critical(f"Manifest config file has blank sections, aborting...")
        exit(1)
    else: 
        if dirs_alr_false:= [
            directory
            for directory in manifest_config['running_experiments']
            if not manifest_config.getboolean('running_experiments', directory)
            and directory in remove_dirs
        ]:
            for dirs in dirs_alr_false:
                LOGGER.info(f"Directory {dirs} has already been removed from execution list, moving on...\n")
        else:
            print(f"Removing the following dirs from execute list:\n")
            print(*remove_dirs, sep=", ")

    ini_files = [
        ini_file
        for dirs in remove_dirs
        for ini_file in (exp_root / dirs).glob("*.ini")
    ]

    for ini_file in ini_files: 
        ini_attr = configparser.ConfigParser()
        ini_attr.read(ini_file)
        ini_attr['run']['execute'] = "false"

        with ini_file.open("w") as file: 
            ini_attr.write(file)     

def add_to_exec(root: str, add_dirs: list, manifest: str):

    exp_root = get_exp_path(root)

    manifest_path = Path(get_testdir()) / manifest
    manifest_config = configparser.ConfigParser()
    manifest_config.read(manifest_path)

    # Run a check to make sure that all of the dirs to be removed actually exist in experiments...
    if bad_dirs := [
        dirs
        for dirs in add_dirs
        if not Path.is_dir(exp_root / dirs)
    ]:
        for dir in bad_dirs:
            LOGGER.error(f"\n\nDirectory '{dir}' doesn't exist in experiment root '{root}', aborting...\n")
        quit()

    if dirs_alr_true := [
        directory
        for directory in manifest_config['running_experiments']
        if manifest_config.getboolean('running_experiments', directory)
        and directory in add_dirs
    ]:
        for dirs in dirs_alr_true: 
            LOGGER.info(f"Directory {dirs} has already been included in the execute list, moving on...\n")
    else: 
        print(f"Adding the following dirs to the execute list:\n")
        print(*add_dirs, sep=", ")

    ini_files = [
        ini_file
        for dirs in add_dirs
        for ini_file in (exp_root / dirs).glob("*.ini")
    ]

    for ini_file in ini_files: 
        ini_attr = configparser.ConfigParser()
        ini_attr.read(ini_file)
        ini_attr['run']['execute'] = "true"

        with ini_file.open("w") as file: 
            ini_attr.write(file)    

def main(): 

    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true", help="Print additional output information")
    parser.add_argument("--debug", action="store_true", help="Print debugging output information, as well as verbose output")
    parser.add_argument("--version", action="store_true", help="Print version info for this file")
    parser.add_argument("-r", "--reset", action="store_true", help="Create new default initialization files for each test directory. Will overwrite any existing .ini files and their contents")
    parser.add_argument("-d", "--dry-run", action="store_true", help="Show manifests without writing changes")
    parser.add_argument("-e", "--exp-root", default="experiments", help="Store the location of the directory housing all available experiments")
    parser.add_argument("-m", "--manifest", default="manifest.ini" , action="store_true", help="Store filename for manifest.ini file") # Kind of redundant, we could just have the manifest always start with all run values being false, then update based on the .ini files in each test dir. 
    parser.add_argument("--remove", nargs='+', help="Store a list of directories to be removed from the running experiments list")
    parser.add_argument("--add-dirs", nargs='+', help="Store a list of directories to be added individually to the running experiments list")
    #parser.add_argument("-f", "--force", action="store_true", help="Overwrite existing .ini files") # Also kind of redundant because -r already works like a force, and all manual changes are going to be written to the manifest regardless...

    args = parser.parse_args()
    #force = args.force

    setup_logger(args.debug, args.verbose)

    LOGGER.debug(f"Printing args: {args}")

    # First step should be to check the validity of root, because all of the methods rely on it...
    if(args.version):
        print(f"\n\nFile {__title__}, version {__version__}\n\n")
        exit(0)

    if (Path.is_dir(get_exp_path(args.exp_root))):
        LOGGER.debug(f"Experiments root '{args.exp_root}' is valid, proceding...\n")
    else: 
        LOGGER.error(f"Experiments root '{args.exp_root}' is invalid. Check run command arguments...\n")
        exit(0)
 
    if args.reset:
        LOGGER.info(f"Resetting experiment .ini files\n\n") 
        build_exp_ini(args.exp_root)
    if args.remove:
        remove_from_exec(args.exp_root, args.remove, args.manifest)
    elif args.add_dirs: 
        add_to_exec(args.exp_root, args.add_dirs, args.manifest)
    if args.dry_run: 
        dry_run(args.exp_root)
        exit(0)

    write_manifest(build_manifest(args.exp_root))

if __name__ == "__main__":
    main()