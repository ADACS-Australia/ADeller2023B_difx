#!/usr/bin/env python3
import os
import re
import subprocess
import argparse
import shutil
import configparser
# make this astropy import conditional with try/except ImportError
from astropy.io import fits 
import warnings
import filecmp
import requests
import itertools
import glob
import resource
import time
import logging 
from pathlib import Path
# from configparser import ConfigParser # For reading config data from initialization files



LOGGER = logging.getLogger(__name__)

def get_testdir():
  # NB: assumes that test data are (/are to be) in the same dir as DiFXtest.py
  filepath = os.path.realpath(os.path.dirname(__file__))
  test_directory = os.path.join(filepath, "experiments")

  if not os.path.isdir(test_directory):
    raise FileNotFoundError(
      f"Test directory does not exist: {test_directory}"
    )
  return test_directory

def check_mpifxcor_installed():
  locate = None
  locate = shutil.which('mpifxcorr')
  if (locate == None):
    print('mpifxcorr not found please make sure DiFX is installed')
    quit()

def check_vdifsim_installed():
  locate = None  # TODO don't need this line because the next line overwrites it
  locate = shutil.which('vdifsim')
  if (locate == None):  # TODO needs to be `is` instead of `==`
    print('vdifsim not found, please make sure you are using a version of DiFX that has vdifsim installed')
    print('DiFXtest will work with most versions of DiFX after the test data has been generated.') 
    quit()

def print_version_info():
  test_directory = get_testdir()
  full_filepath = test_directory + "/difx_version_info.txt"
  print()
  with open(full_filepath,"r") as file:
    contents = file.read()
    print(contents)
  difx_version = os.environ['DIFX_VERSION']
  print("CURRENTLY USING VERSION: " + difx_version);
  print()

def record_difx_version_info():
  test_directory = get_testdir()
  full_filepath = test_directory + "/difx_version_info.txt"
  difx_version = os.environ['DIFX_VERSION']
  with open(full_filepath,"w") as fh:  # TODO always use `with open() as fh: ...`
    filecontents = "BENCHMARK CREATED WITH DIFX_VERISON: " + difx_version + "\n"  
    fh.write(filecontents)
    fh.close()

def generate_vdifsim_config(cores):
  filepath = os.path.expanduser('~') 
  #print(filepath)
  filename = filepath + "/.vdifsim" 
  fh = open(filename,"w")

  core_list = cores.split(",")

  filecontents = "# Processing computers\n"
  for item in core_list:
    filecontents = filecontents + item + "\n"

  current_directory = get_testdir()

  LOGGER.debug(f"current_directory in generate_vdifsim_config: {current_directory}")

  filecontents = filecontents + "\n# data destinations\n"
  filecontents = filecontents + "BR " + current_directory + "/testdata/ localhost\n"
  filecontents = filecontents + "FD " + current_directory + "/testdata/ localhost\n"
  filecontents = filecontents + "HN " + current_directory + "/testdata/ localhost\n"
  filecontents = filecontents + "KP " + current_directory + "/testdata/ localhost\n"
  filecontents = filecontents + "LA " + current_directory + "/testdata/ localhost\n"
  filecontents = filecontents + "MK " + current_directory + "/testdata/ localhost\n"
  filecontents = filecontents + "NL " + current_directory + "/testdata/ localhost\n"
  filecontents = filecontents + "OV " + current_directory + "/testdata/ localhost\n"
  filecontents = filecontents + "PT " + current_directory + "/testdata/ localhost\n"
  filecontents = filecontents + "SC " + current_directory + "/testdata/ localhost\n"
  

  fh.write(filecontents)
  fh.close()
  

# For now just set threads to 1 per machine to get repeatable results 
def set_numthreads(testname,numthreads):

  current_directory = get_testdir()
  working_directory = current_directory + "/" + testname + "/"
  v2d_filename = working_directory + testname + ".v2d"
  #print(v2d_filename)
  with open(v2d_filename,'r') as file:
    content = file.read()
    nthreads_string = "nThread = " + numthreads
    content = content.replace("nThread = 12",nthreads_string);
  with open(v2d_filename,'w') as file:
    file.write(content)

def generate_v2d(testname):


  current_directory = get_testdir()
  working_directory = current_directory + "/" + testname + "/"
  testdata_dir = current_directory + "/testdata"

  args = "oms2v2d -v -v --sim " + testname + ".oms "
  logfile = working_directory + "/oms2v2d.log"
  errfile = working_directory + "/oms2v2d.err"
  log = open(logfile,"w")
  err = open(errfile,"w")

  #vdifsim_environment = read_path()


  #print(vdifsim_environment)
  
  try:
    proc = subprocess.Popen(args,cwd=working_directory,shell=True,stdout=log,stderr=err) # former env: {"PATH":vdifsim_environment}
    proc.wait()
    LOGGER.debug("generated v2d")
  #  print(working_directory + testname + ".v2d")
  #  print(os.path.exists(working_directory + testname + ".v2d"))
  except subprocess.CalledProcessError as e:
    LOGGER.error("oms2v2d failed check log files:\n")
    raise(e)
   
  return


# Adds sampling key to v2d file to generate complex data (FD set to complex, HN set to real) 
def add_complex_sampling_to_v2d(testname):
  current_directory = get_testdir()
  working_directory = current_directory + "/" + testname + "/"
  v2d_filename = working_directory + testname + ".v2d"
  #print(v2d_filename)
  with open(v2d_filename,'r') as file:
    content = file.read()
    station_sampling_string_FD = "ANTENNA FD { format=VDIFC/5032/2 " 
    content = content.replace("ANTENNA FD { ",station_sampling_string_FD);
  with open(v2d_filename,'w') as file:
    file.write(content)

def add_complex_sampling_to_v2d_all_complex(testname):
  current_directory = get_testdir()
  working_directory = current_directory + "/" + testname + "/"
  v2d_filename = working_directory + testname + ".v2d"
  #print(v2d_filename)
  with open(v2d_filename,'r') as file:
    content = file.read()
    station_sampling_string_FD = "ANTENNA FD { format=VDIFC/5032/2 "
    content = content.replace("ANTENNA FD { ",station_sampling_string_FD);
    station_sampling_string_FD = "ANTENNA HN { format=VDIFC/5032/2 "
    content = content.replace("ANTENNA HN { ",station_sampling_string_FD);
  with open(v2d_filename,'w') as file:
    file.write(content)


def generate_filelist(testname):
  LOGGER.debug(f"started generate filelist")
  filepath = get_testdir()

  current_directory = get_testdir()
  working_directory = current_directory + "/" + testname + "/"
  testdata_dir = current_directory + "/testdata"
  #logger.debug(f" testdata_dir: {testdata_dir}")
  LOGGER.debug(f"testname: {testname}")
  #vdifsim_environment = read_path()

  if (testname[-3:] == "gpu"):
    basename = testname[:-4]  
    #print("Basename: " + basename)
    args = "makesimfilelist -f" + " --name=" + basename + " " + testname + ".vex " + testdata_dir
    LOGGER.debug(f"makesimfilelist subproc call: {args}")
    try: 
      proc = subprocess.Popen(args,cwd=working_directory,shell=True,stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT)
      LOGGER.debug(f"makesimfilelist outcode: {proc.wait()}")
      #
    except subprocess.CalledProcessError as e: 
      print(f"Error with makesimfilelist call : \n{e}")
#subprocess.call(args,cwd=working_directory,shell=True,stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT)
    contents = os.listdir(working_directory)
    for item in contents:
      if (item[-9:] == ".filelist"):
        #logger.debug(f" filelist name before change: {item}")
        filelist_name = working_directory + item  
        str1 = testname[:-4] + "."
        str2 = testname + "." 
        new_filelist_name = filelist_name.replace(str1,str2)
        os.rename(filelist_name,new_filelist_name)
        #logger.debug(f" filelist name after change: {item}")
  else: 
    args = "makesimfilelist -f " + testname + ".vex " + testdata_dir
    #print(testdata_dir)
    try: 
      proc = subprocess.Popen(args,cwd=working_directory,shell=True,stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT)
      LOGGER.debug(f"makesimfilelist outcode: {proc.wait()}")
      #print(f"Error output: {proc.stderr}")
    except subprocess.CalledProcessError as e: 
      LOGGER.error(f"Error with makesimfilelist call : \n{e}")
    #print("IMHERE: generated filelist(s)")
    #print(proc.stderr)
  check_for_filelists(testname)
  check_testdata_dir(testname)
  
  return 

def check_for_filelists(testname): 
  current_directory = get_testdir()
  working_directory = current_directory + "/" + testname + "/"
  contents = os.listdir(working_directory)
  LOGGER.debug(f"checking for filelists...")
  for item in contents: 
    if(item[:-9] == ".filelist"): 
      LOGGER.debug(f"found filelist: {item}")
  return 

def check_testdata_dir(testname):

  # Check /testdata for each .sim file under testname

  basename = ""

  if(testname[-4:] == "-gpu"):
    basename = testname[:-4]
  else: 
    basename = testname
  current_directory = get_testdir()
  testdata_dir = current_directory + "/testdata"
  #print("testdata_dir: " + testdata_dir)
  #print("testname: " + testname)

  contents = os.listdir(testdata_dir)
  for item in contents: 
      if (item[:-14] == basename.upper()):
        LOGGER.debug(f"file size for {item}: {str(os.path.getsize(testdata_dir + '/' + item))}")


  return

def get_results_and_setup_files():
  current_directory = get_testdir()
  tarball = current_directory + "/tests.tgz"
  testdata_dir = current_directory + "/testdata"
  
  

  if (os.path.isdir(testdata_dir) == False):
    testdata_dir = current_directory + "/testdata"
    os.mkdir(testdata_dir)

  #print(tarball)
  if (os.path.isfile(tarball)):
    return
  else:
    url = "https://github.com/difx/difx-data/raw/main/tests.tgz"
    try:
      arg = "wget " + url
      subprocess.call(arg,cwd=current_directory,shell=True,stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT)
    #r = requests.get(url, allow_redirects=True)
    #filename = current_directory + "/tests.tgz"
    #open(filename,'wb').write(r.content)
    except:
      print("Error downloading setup files and initial benchmark results")


def filter_auto_correlations(fits_filename):

  print("Filtering auto correlations.")
  warnings.resetwarnings() 
  warnings.filterwarnings('ignore',category=UserWarning,append=True)  
  warnings.filterwarnings('ignore', category=RuntimeWarning, append=True)
  hdulist = fits.open(fits_filename)
  tbdata = hdulist[8].data
  hdulist.close()  


  mask = (tbdata['BASELINE'] % 257) != 0
  new_tbdata = tbdata[mask] 

  fits.update(fits_filename,new_tbdata,8)
  hdulist = fits.open(fits_filename)
  tbdata = hdulist[8].data
  hdulist.close()
  warnings.resetwarnings()
  warnings.filterwarnings('always',category=UserWarning,append=True)
  warnings.filterwarnings('ignore', category=RuntimeWarning, append=True)


def rename_oms_v2d(testname): 
  current_directory = get_testdir()
  working_directory = current_directory + "/" + testname + "/"
  contents = os.listdir(working_directory)

  arg = ""
  old_filename = ""
  new_filename = ""
  
  for item in contents: 
    if (item[-4:] == '.vex' or item[-4:] == '.oms'):
      if(item[:-4] != testname):

        LOGGER.debug(f" renaming oms/v2d file: {item}")
        
        old_filename = item
        new_filename = testname + item[-4:]
        arg = "mv " + old_filename + " " + new_filename
        #print(old_filename)
        #print(new_filename)
        #print(arg)
        
        try:
          proc=subprocess.Popen(arg,cwd=current_directory,shell=True,stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT)
          proc.wait()
          LOGGER.debug(f" renamed oms/v2d for {item}")
        except subprocess.CalledProcessError as e:
          print("Error with changing names of base .oms and .vex files to match eperiment:")
          raise(e)

      # If this method is activated, the filelist basenames in the .v2d file will automatically match the basename of the experiment directory, causing vex2difx to crash because there are no "-GPU" filelists. 
      # From here, you need to make a subproc call that seds the expected filelist string in the .v2d file to the original experiment basename (without "-gpu"). 
      # Don't have to worry about makesimfilelist, because it has the proper basename baked into the subproc command arg. 

      # Subproc regex: 

def align_chans_ffts(testname): 

  test_dir = get_testdir() # DiFXtest_vdifsim

  # Method to resolve discrepancy between num channels and ffts across .oms and .v2d files between gpu and cpu filesets
  # Search for any discrepancy, and resolve the values to cpu values if found
  # We are looking for number_channels in the .vex file and FFT_size in .oms file 
  # We are also constraining this to only run in -gpu directories within the main method

  cpu_testname = testname[:-4]
  cpu_test_dir = test_dir + "/" + cpu_testname + "/"
  cpu_vex = cpu_test_dir + cpu_testname + ".vex"
  cpu_oms = cpu_test_dir + cpu_testname + ".oms"
  LOGGER.debug(f"Checking cpu_vex: {cpu_vex}, path exists: {os.path.exists(cpu_vex)}\nChecking cpu_oms: {cpu_oms}, path exists: {os.path.exists(cpu_oms)}")

  gpu_test_dir = test_dir + "/" + testname + "/"
  gpu_vex = gpu_test_dir + testname + ".vex"
  gpu_oms = gpu_test_dir + testname + ".oms"
  LOGGER.debug(f"Checking gpu_vex: {gpu_vex}, path exists: {os.path.exists(gpu_vex)}\nChecking gpu_oms: {gpu_oms}, path exists: {os.path.exists(gpu_oms)}")

  # Grabbing correct number_channels from cpu *.vex
  try: 
    result = subprocess.run(
    ["grep", "-m", "1", r"^[[:space:]]*\*[[:space:]]*number_channels[[:space:]]*:", cpu_vex], 
    capture_output=True, text=True, check=True
    )
    cpu_channels = result.stdout.strip()
    LOGGER.debug(f"Checking cpu_channels: {cpu_channels}")
  except subprocess.CalledProcessError as e:
    LOGGER.error(f"Subproc grep for cpu_channel fix mishandled: \n{e}\n")

  # Replacing number_channels in gpu *.vex
  try: 
    subprocess.run(
    ["sed", "-i",
     f"s|^[[:space:]]*\\*[[:space:]]*number_channels[[:space:]]*:.*$|{cpu_channels}|",
     gpu_vex],
    check=True)
  except subprocess.CalledProcessError as e:
    LOGGER.error(f"Subproc call for replacing number_channels in {testname} *.vex file mishandled: \n{e}")

  # Grabbing correct FFT_size from cpu *.oms
  try: 
    result = subprocess.run(
    ["grep", "-m", "1", r"^[[:space:]]*FFT_SIZE[[:space:]]*=", cpu_oms],
    capture_output=True, text=True,
    )
    cpu_FFTs = result.stdout.strip()
    LOGGER.debug(f"Checking cpu_FFTs: {cpu_FFTs}")
  except subprocess.CalledProcessError as e:
    LOGGER.error(f"Subproc grep for FFT_size fix mishandled: \n{e}")

  # Replacing FFT_SIZE in gpu *.oms
  try: 
    subprocess.run(
    ["sed", "-i", f"s|^[[:space:]]*FFT_SIZE[[:space:]]*=.*$|{cpu_FFTs}|", gpu_oms],
    check=True)
  except subprocess.CalledProcessError as e:
    LOGGER.error(f"Subproc sed for replacing FFT_size in {testname} .vex file mishandled: \n{e}")



def fix_filelist_path(testname):
  current_directory = get_testdir()
  working_directory = current_directory + "/" + testname + "/"
  v2d_files = glob.glob(f"{testname}/*.v2d")

  #grep_pattern = "grep -q -- '-gpu.*.filelist' " + testname + "/*.v2d" 
  #sed_pattern = "sed -q 's/-gpu\.[^.]*\.filelist/.filelist/' " + testname + "/*.v2d"


  try: 
    proc = subprocess.run(["grep", "-q", "--", r"[^[:space:]]*-gpu\.[^.[:space:]]*\.filelist", *v2d_files], check=False).returncode == 0

    if proc: 
      subprocess.run(["sed", "-i",
                      r"s/\([^[:space:]]*\)-gpu\(\.[^.[:space:]]*\.filelist\)/\1\2/g",
                      *v2d_files], check=True) 
    print("IMHERE: fixed filelist path(s)")
  except subprocess.CalledProcessError as e: 
    print(f"Grep to sed for GPU filelist in .v2d failed: {e}")


def report_zero_output(testname):
  # Meant to help notify the user if any of the important final filetypes are zero size
  
  current_directory = get_testdir()
  working_directory = current_directory + "/" + testname + "/"
  #contents = os.listdir(working_directory)

  difx_path = working_directory + "_1.difx"

  if os.path.exists(difx_path):
    contents = os.listdir(difx_path)

    for file in contents: 
      if os.path.getsize(file) == 0: 
        print("!!!" + file + " has a zero byte size on output !!!")

  return 1

def get_binary_files(directory):
  input_files = []
  contents = os.listdir(directory)

  difx_dirs = []
  for item in contents:
    if (item[-5:] == '.difx'):
      difx_dirs.append(item)
    if (item[-6:] == '.input'):
      path_and_input_file = directory + "/" + item
      input_files.append(path_and_input_file)
  difx_binary_files = []
  for difx_dir in difx_dirs:
    difx_dir = directory + "/" + difx_dir
    difx_dir_contents = os.listdir(difx_dir)
    for item in difx_dir_contents:
      if (item[:4] == 'DIFX'):
        path_and_file = difx_dir + "/" + item
        difx_binary_files.append(path_and_file)
  return difx_binary_files,input_files


def get_fits_file(directory):
  contents = os.listdir(directory)
  files_and_paths = list()
  for item in contents:
    if (item[-5:] == '.FITS'):
      file_and_path = directory + item
      files_and_paths.append(file_and_path)
  if (len(files_and_paths) == 1):
    return(files_and_paths[0])
  else:
    return(files_and_paths)


def get_im_file(directory):
  contents = os.listdir(directory)
  for item in contents:
    if (item[-3:] == '.im'):
      file_and_path = directory + item
      return file_and_path

def get_input_files(directory):
    contents = os.listdir(directory)
    input_files = [] 
    for item in contents:
      if (item[-6:] == '.input'):
        input_files.append(item)
    return(input_files)

def get_machine_files(directory):
    contents = os.listdir(directory)
    input_files = []
    for item in contents:
      if (item[-9:] == '.machines'):
        input_files.append(item)
    return(input_files)

def get_output_dirs(directory):
   contents = os.listdir(directory)
   output_dirs = [] 
   for item in contents:
     if (item[-5:] == '.difx'):
       output_dirs.append(item)
   #print(output_dirs)
   return(output_dirs)

def rm_output_files(testname):
  current_directory = get_testdir()
  working_directory = current_directory + "/" + testname + "/"
  contents = os.listdir(working_directory)

  extensions = (".vex", ".oms", ".config", ".ini", ".threads")

  for item in contents:
      #logger.debug(f"Item type in rm_output_files: item: {item}, {type(item)}")
      if (not item.endswith(extensions) 
          and item != 'benchmark_results' 
        ):
        fp_item = working_directory + item 
        if (item[-5:] == '.difx'):
          #print(fp_item)
          shutil.rmtree(fp_item)
        else:
          os.remove(fp_item)
  #quit()


def run_vex2difx(testname):
  current_directory = get_testdir()
  working_directory = current_directory + "/" + testname + "/"
  vex2difxlogfile = working_directory + "/vex2difx.log"
  vex2difxerrfile = working_directory + "/vex2difx.error"
  f_vex2difxlog = open(vex2difxlogfile,"w")
  f_vex2difxerr = open(vex2difxerrfile,"w")

  arg = "vex2difx -f " + testname + ".v2d"

  try:
    proc = subprocess.Popen(arg,cwd=working_directory,shell=True,stdout=f_vex2difxlog,stderr=f_vex2difxerr)
    LOGGER.debug(f"vex2difx process outcode: {proc.wait()}")
  except subprocess.CalledProcessError as e:
    LOGGER.error(f"vex2difx failed check log files:\n")
    #print(vex2difxlogfile)
    #print(vex2difxerrfile)
    #print()
    raise(e)

def run_difxcalc(testname):
  current_directory = get_testdir()
  working_directory = current_directory + "/" + testname + "/"
  difxcalclogfile = working_directory + "/difxcalc.log"
  difxcalcerrfile = working_directory + "/difxcalc.error"
  f_difxcalclog = open(difxcalclogfile,"w")
  f_difxcalcerr = open(difxcalcerrfile,"w")

  if testname in (
    "rdv70" , "v252f" , "rdv70-gpu" , "v252f-gpu"
  ): 
    arg = "difxcalc *.calc"
  else:
    arg = "difxcalc *.calc" 

  proc = subprocess.run(arg,cwd=working_directory,shell=True,stdout=f_difxcalclog,stderr=f_difxcalcerr,check=True)
  contents = os.listdir(working_directory)
  im_filename = "" 
  for item in contents:
    if (item[-3:] == ".im"):
      im_filename = item

  if not im_filename:
    print("difxcalc failed check log files:")
    print(difxcalclogfile)
    print(difxcalcerrfile)
    quit()

def rm_test_data():
  print("removing test data")
  testdir = get_testdir()
  testdata_dir = testdir + "/testdata"
  contents = os.listdir(testdata_dir)
  for item in contents:
    
    file_path = os.path.join(testdata_dir,item)
    os.unlink(file_path)

def record_path():
  path_value = os.environ.get('PATH')
  current_directory =  get_testdir()
  tmp_dir = current_directory + "/.tmp"
  if (os.path.isdir(tmp_dir) == False):
     os.mkdir(tmp_dir)
  full_filepath = tmp_dir + "/path.txt"
  fh = open(full_filepath,"w") 
  fh.write(path_value)
  fh.close()

def read_path():
  current_directory =  get_testdir()
  tmp_dir = current_directory + "/.tmp"
  full_filepath = tmp_dir + "/path.txt"
  fh = open(full_filepath,"r")
  path = fh.read()
  return path

def run_vdifsim(testname):

  seed = "7276"   
 
  current_directory =  get_testdir()
  working_directory = current_directory + "/" + testname + "/"
  config_file = working_directory + "src_ant.config"
  vdifsimlogfile = working_directory + "/vdifsim.log"
  vdifsimerrfile = working_directory + "/vdifsim.error"
  f_vdifsimlog = open(vdifsimlogfile,"w")
  f_vdifsimerr = open(vdifsimerrfile,"w")

  arg = "vdifsim -s " + seed + " -c " + config_file + " "  + testname + "_*.input"
  #print(arg)

  try:
    proc = subprocess.run(arg,cwd=working_directory,shell=True,stdout=f_vdifsimlog,stderr=f_vdifsimerr,check=True,text=True,timeout=7)
    print("run_vdifsim return code: " + str(proc.returncode))
  except subprocess.CalledProcessError as e:
    print("vdifsim failed check log files: ")
    print(vdifsimlogfile)
    print(vdifsimerrfile)
    raise(e)
  except subprocess.TimeoutExpired as t:
    print("\n########### TIMEOUT DETECTED ###########")
    print(f"\nCommand run: {t.cmd}")
    print(f"\nvdifsim command has hung for more than {t.timeout} seconds. Check creation process for vdif/filelist sim data...\n" )
    return bool(t.timeout)
  
  return False

def run_mpifxcorr(testname, numcores):
  
  print(f"Running test {testname}")
  cpu_time = 0
  real_time = 0

  current_directory = get_testdir() 
  working_directory = current_directory + "/" + testname + "/"
  
  input_files = get_input_files(working_directory)
  machine_files = get_machine_files(working_directory)

  for (machine_file, input_file) in zip(machine_files, input_files):
    arg = "mpirun -machinefile " + machine_file + " -np " + str(numcores) + " mpifxcorr " + input_file
    LOGGER.debug(arg)
    difxlogfile = working_directory + "/mpifxcorr.log"
    difxerrfile = working_directory + "/mpifxcorr.error"
    f_difxlog = open(difxlogfile,"w")
    f_difxerr = open(difxerrfile,"w")
    try:
      usage_start = resource.getrusage(resource.RUSAGE_CHILDREN)
      t1 = time.perf_counter()
      proc = subprocess.run(arg,cwd=working_directory,shell=True,stdout=f_difxlog,stderr=f_difxerr,check=True)
      LOGGER.debug(f"Working directory: {working_directory}")
      t2 = time.perf_counter()
      usage_end = resource.getrusage(resource.RUSAGE_CHILDREN)
      cpu_time = usage_end.ru_utime - usage_start.ru_utime
      real_time = t2-t1 
      print('Correlation complete')
      print(f'CPU processing time:   {cpu_time}')
      print(f'Real run time:         {real_time}')
      print()
    except subprocess.CalledProcessError as e:
      print("mpifxcorr failed check log files:")
      print(difxlogfile)
      print(difxerrfile) 
      print()
      return  
    # Add timeout error for when mpifxcorr gets hung when restarting tests

def run_mpifxcorr_gpumode(testname, numcores):
  
  print("Running test " + testname)

  cpu_time = 0
  real_time = 0


  current_directory = get_testdir()
  working_directory = current_directory + "/" + testname + "/"

  input_files = get_input_files(working_directory)
  machine_files = get_machine_files(working_directory)

  for (machine_file, input_file) in zip(machine_files, input_files):
    arg = "mpirun -machinefile " + machine_file + " -np " + str(numcores) + " mpifxcorr --usegpu " + input_file

    LOGGER.debug(f"Run_mpifxcorr_gpumode run command: {arg}")
    difxlogfile = working_directory + "/mpifxcorr.log"
    difxerrfile = working_directory + "/mpifxcorr.error"
    f_difxlog = open(difxlogfile,"w")
    f_difxerr = open(difxerrfile,"w")
    try:
      usage_start = resource.getrusage(resource.RUSAGE_CHILDREN)
      t1 = time.perf_counter()
      proc = subprocess.run(arg,cwd=working_directory,shell=True,stdout=f_difxlog,stderr=f_difxerr,check=True)
      t2 = time.perf_counter()
      usage_end = resource.getrusage(resource.RUSAGE_CHILDREN)
      cpu_time = usage_end.ru_utime - usage_start.ru_utime
      real_time = t2-t1
      print('Correlation complete')
      print(f'CPU processing time:   {cpu_time}')
      print(f'Real run time:         {real_time}')
      print()
    except subprocess.CalledProcessError as e:
      print("mpifxcorr failed check log files:")
      print(difxlogfile)
      print(difxerrfile)
      print()
      return

def run_difx2fits(testname,filterautocorrs):

  testdir = get_testdir()
  working_directory = testdir + "/" + testname

  difx2fitslogfile = working_directory + "/difx2fits.log"
  difx2fitserrfile = working_directory + "/difx2fits.error"
  f_difx2fitslog = open(difx2fitslogfile,"w")
  f_difx2fitserr = open(difx2fitserrfile,"w")

  output_dirs = get_output_dirs(working_directory)
  
  outputfile_name = testname + ".FITS"
  if (filterautocorrs == "YES"):
    outputfile_name = testname + "-filtered.FITS"
  else:
    outputfile_name = testname + ".FITS"

  for item in output_dirs:
    arg = "difx2fits " + item[:-5] + " " + outputfile_name   
    #print(arg)
    try:
      proc = subprocess.run(arg,cwd=working_directory,shell=True,stdout=f_difx2fitslog,stderr=f_difx2fitserr,check=True)
    except subprocess.CalledProcessError as e:
      print("difx2fits failed check log files:")
      print(difx2fitslogfile)
      print(difx2fitserrfile)
      print()
      raise()
    
def is_testdata_empty(testname):
  testdir = get_testdir()
  testdata_dir = testdir + "/testdata"
  testdata_root = testname.upper()
  testdata_root = testdata_dir + "/" + testdata_root + "*"
  testdata_files =  glob.glob(testdata_root)
  if (len(testdata_files) == 0):
    return True
  else:
    return False

def compare_results(testname, abstol, reltol, filterautocorrs):

  LOGGER.info(f"Comparing results for test {testname}")

  # results list [Binary file same as benchmark?,FITS file same as benchmark?]
  results = ["",True]

  current_directory = get_testdir()
  working_directory = current_directory + "/" + testname + "/"   
  #print(working_directory)
  binfiles,inputfiles = get_binary_files(working_directory)
  #print("*******")
  #print(binfiles)

  # The binary files don't even exist
  if (len(binfiles) == 0):
    LOGGER.info(f"Zero length binary files, aborting results comparison...\n")
    results[0] = "FAILED"
    results[1] = False
    return(results)

  binfile_stat = os.stat(binfiles[0])
  binfile_size = binfile_stat.st_size
  #print(binfile_size)
  if (binfile_size == 0): 
    LOGGER.critical(f"Zero size binary files, aborting results comparison...\n")
    results[0] = "FAILED"
    results[1] = False
    return(results)

  binfile = binfiles[0] 
  inputfile = inputfiles[0]
  results_directory = working_directory + "/benchmark_results/" 
  binfiles2,inputfiles2 = get_binary_files(results_directory) 
  benchmark_binfile = binfiles2[0]

  try: 
    arg = "diffDiFX.py " + binfile + " " + benchmark_binfile + " " + "-i " + inputfile + " > binary_diff.log"   
    #print("Arg: " + arg)
    proc = subprocess.Popen(arg,cwd=working_directory,shell=True) 
    LOGGER.debug(f"diffDiFX.py outcode: {proc.wait()}")
  except subprocess.CalledProcessError as zd: 
    LOGGER.error(f"ZeroDivision in diffDiFX.py subproc call: {zd}")


  fits_file = get_fits_file(working_directory)
  fits_file_benchmarks = get_fits_file(results_directory)
  #print(fits_file_benchmarks)
  if (filterautocorrs == "YES"):
    for item in fits_file_benchmarks: 
      if (item[-13:] == "filtered.FITS"):
        fits_file_benchmark = item
  else:
    for item in fits_file_benchmarks:
      if (item[-13:] != "filtered.FITS"):
        fits_file_benchmark = item

  #print(fits_file)
  #print(fits_file_benchmark)
  #quit()



  # filter autocorrelations out of fits file
  #print(fits_file)
  if (filterautocorrs == "YES"):
    filter_auto_correlations(fits_file)
    
  hdu1 = fits.open(fits_file)
  hdu2 = fits.open(fits_file_benchmark) 

  warnings.resetwarnings()
  warnings.filterwarnings('ignore', category=UserWarning, append=True) 

  fd = fits.FITSDiff(hdu1, hdu2, ignore_keywords=['DATE-MAP','CDATE','HISTORY'], atol=abstol , rtol=reltol)  
  diff_fitslog = working_directory + "/fits_diff.log"  
  output = fd.report(fileobj=diff_fitslog, indent=0, overwrite=True) 
  results[1] = fd.identical
  hdu1.close()
  hdu2.close()

  fp_binary_diff_file = working_directory + 'binary_diff.log'
  #print(fp_binary_diff_file)
  with open(fp_binary_diff_file, encoding="utf8", errors='ignore') as f:
  #with open(fp_binary_diff_file) as f:
    last_line = f.readlines()[-1]
  last_line = str(last_line)
  last_line = last_line.strip('\n')
  results[0] = last_line

  print() 
  return(results)

def compare_results_gpu_v_cpu(testname, abstol, reltol, filterautocorrs):

  LOGGER.info(f"Comparing results for test {testname}")

  # results list [Binary file same as benchmark?,FITS file same as benchmark?]
  results = ["",True]

  current_directory = get_testdir()
  
  #usb-gpu-vs-cpu

  substring = "-vs-cpu" 
  testname_gpu = testname.replace(substring,"")  
  substring = "-gpu-vs-cpu"  
  testname_cpu = testname.replace(substring,"")
  
  working_directory = current_directory + "/" + testname_gpu + "/"
  #print(working_directory)
  binfiles,inputfiles = get_binary_files(working_directory)

  if (len(binfiles) is 0):
    print(f"Binary files not created, aborting comparison...\n")
    results[0] = "FAILED"
    results[1] = False
    print() # Zero binary file scenario for cpu-vs-gpu tests
    return(results)

  binfile = binfiles[0] 
  inputfile = inputfiles[0]

  binfile_stat = os.stat(binfile)
  binfile_size = binfile_stat.st_size

  if (binfile_size is 0):
    LOGGER.critical(f"Zero size binary files, aborting results comparison...\n")
    results[0] = "FAILED"
    results[1] = False
    print()
    return(results)



  #print(binfiles)
 
  results_directory = current_directory + "/" + testname_cpu + "/benchmark_results/"
  binfiles2,inputfiles2 = get_binary_files(results_directory) 
  benchmark_binfile = binfiles2[0]

  LOGGER.debug(f"Working directory: {working_directory}")
  LOGGER.debug(f"Benchmark binfile: {benchmark_binfile} path exits: {os.path.exists(benchmark_binfile)}")
  LOGGER.debug(f"Binfile: {binfile} path exists: {os.path.exists(binfile)}")
  LOGGER.debug(f"Binfile size: {os.path.getsize(binfile)}")
 

  # Check for zero file size in any failed test directories. 
  
# .ini

  if(os.path.getsize(binfile) == 0):
    LOGGER.critical(f"DiFX binary file {binfile} has zero size, aborting diffDiFX call...")
  else: 
    try: 
      #print(f"Running diffDiFX.py in {testname}")
      arg = "diffDiFX.py " + binfile + " " + benchmark_binfile + " " + "-i " + inputfile + " > binary_diff_vs_cpu.log"   
      LOGGER.debug(f"diffDiFX.py call: {arg}")
      proc = subprocess.Popen(arg,cwd=working_directory,shell=True)
      LOGGER.debug(f"diffDiFX.py outode: {proc.wait()}")
    except subprocess.CalledProcessError as e:
      LOGGER.error(f"diffDiFX.py call failed: \n")

    
  fits_file = get_fits_file(working_directory)


  fits_file_benchmarks = get_fits_file(results_directory)
  if (filterautocorrs == "YES"):
    for item in fits_file_benchmarks:
      if (item[-13:] == "filtered.FITS"):
        fits_file_benchmark = item
  else:
    for item in fits_file_benchmarks:
      if (item[-13:] != "filtered.FITS"):
        fits_file_benchmark = item


  # filter autocorrelations out of fits file
  if (filterautocorrs == "YES"):
   filter_auto_correlations(fits_file)
 
 # print()
 # print(fits_file)
 # print(fits_file_benchmark)
 # print() 

  hdu1 = fits.open(fits_file)
  hdu2 = fits.open(fits_file_benchmark) 

  warnings.resetwarnings()
  warnings.filterwarnings('ignore', category=UserWarning, append=True) 
  warnings.filterwarnings('ignore', category=RuntimeWarning, append=True)
  fd = fits.FITSDiff(hdu1, hdu2, ignore_keywords=['DATE-MAP','CDATE','HISTORY'], atol=abstol , rtol=reltol)  
  diff_fitslog = working_directory + "/fits_diff_vs_cpu.log"  
  output = fd.report(fileobj=diff_fitslog, indent=0, overwrite=True) 
  results[1] = fd.identical
  hdu1.close()
  hdu2.close()
  warnings.resetwarnings()
  warnings.filterwarnings('always', category=UserWarning, append=True)
  warnings.filterwarnings('always', category=RuntimeWarning, append=True)
  #print("comparing fits_file = " + fits_file + " VS. benchmark fits file = " + fits_file_benchmark)
  #print(fd.identical)
  #print(output)

  fp_binary_diff_file = working_directory + 'binary_diff_vs_cpu.log'
  #print(fp_binary_diff_file)
  with open(fp_binary_diff_file, encoding="utf8", errors='ignore') as f:
  #with open(fp_binary_diff_file) as f:
    last_line = f.readlines()[-1]
  last_line = str(last_line)
  last_line = last_line.strip('\n')
  results[0] = last_line

  print()
  return(results)



def compare_im_files(testname):
  current_directory = get_testdir()
  working_directory = current_directory + "/" + testname + "/"
  results_directory = working_directory + "/benchmark_results/"

  im_file = get_im_file(working_directory)
  im_file_benchmark = get_im_file(results_directory)

  if (im_file == None):
    return

  if (im_file_benchmark == None):
    return  

  if (filecmp.cmp(im_file,im_file_benchmark) == False):
    output = "Warning model file " + im_file + " differs from benchmark. difxcalc version could differ from benchmark."
    print(output)


def update_test(testname, filterautocorrs):
  print("Updating test " + testname)

  current_directory = get_testdir()
  working_directory = current_directory + "/" + testname + "/"
  results_directory = working_directory + "/benchmark_results/" 
  binfiles,inputfiles = get_binary_files(working_directory)

  # skip if difx has not run succesfully
  if len(binfiles)==0:
    return

  binfile_stat = os.stat(binfiles[0])
  binfile_size = binfile_stat.st_size
  if (binfile_size < 1):
    return
  
  current_binfile = binfiles[0]

  binfiles,inputfiles = get_binary_files(results_directory)
  results_binfile = binfiles[0]

  
  #print(current_binfile)  

  fits_file = get_fits_file(working_directory)
  fits_file_benchmarks = get_fits_file(results_directory)
  if (filterautocorrs == "YES"):
    for item in fits_file_benchmarks:
      if (item[-13:] == "filtered.FITS"):
        fits_file_benchmark = item
  else:
    for item in fits_file_benchmarks:
      if (item[-13:] != "filtered.FITS"):
        fits_file_benchmark = item

  #print()
  #print(fits_file)


  #print(fits_file_benchmark)
  #print(current_binfile)
  os.remove(fits_file_benchmark)
  os.remove(results_binfile)
  args = "cp " + fits_file + " " + fits_file_benchmark
  subprocess.call(args,shell=True)
  #shutil.copy2(fits_file,fits_file_benchmark)
  shutil.copy2(current_binfile,results_binfile) 

  im_file = get_im_file(working_directory)
  im_file_benchmark = get_im_file(results_directory)
  os.remove(im_file_benchmark) 
  shutil.copy2(im_file,im_file_benchmark)

def repackage_tests(testnames):
  LOGGER.info(f"Repacking tarball...")
  current_directory = get_testdir()

  #logger.debug(f"Prining current_directory: {current_directory}")

  tar_command = "tar -zcvf " + current_directory + "/tests.tgz difx_version_info.txt"
   
  for testname in testnames:
     
    working_directory = current_directory + "/" + testname + "/"
    contents = os.listdir(working_directory) 
    for item in contents:
      if (
        item.endswith(".vex")
        or item.endswith(".oms")
        or item.endswith(".config")
        or item.endswith(".ini")
        or item == "benchmark_results"
      ):     
        #print(item)
        if (item == 'benchmark_results'):
          tar_command = tar_command + " " + testname + "/" + "benchmark_results/*" 
        else:
          tar_command = tar_command + " " + testname + "/" + item
    #end for loop
  #logger.debug(f"Printing tar command at the end of repack_tests:\n\n{tar_command}")
  proc = subprocess.run(tar_command,cwd=current_directory,shell=True, capture_output=True, text=True)
  #logger.debug(f"Tar_command output: {proc.stdout}")


def unpackage_tests(testnames):

  unpack = False
  current_directory = get_testdir()
  for testname in testnames:
    testname = current_directory + "/" + testname
    if (os.path.isdir(testname) == False):
      unpack = True
  if (unpack):
    print("Unpacking test tarball.") 
    subprocess.call("tar -zxvf tests.tgz",shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

def create_test_data(testname):
 
  print("Creating test data")

def set_manifest(root, reset):
  cmd = "./runconfig -v -e " + os.path.basename(root)

  if reset: cmd += " -r"

  LOGGER.debug(f"Set_manifest command: {cmd}")

  try: 
    proc = subprocess.run(cmd, cwd=get_testdir(), shell=True, capture_output=True, text=True)
    LOGGER.debug(f"Set_manifest command output: {proc.stdout}")
  except subprocess.CalledProcessError as e:
    LOGGER.error(f"Issue with set_manifest method: {e}")

def create_manifest(root):
  cmd = "./runconfig -v -e " + os.path.basename(root)
  
  try:
    proc = subprocess.run(cmd, cwd=get_testdir(), shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
  except subprocess.CalledProcessError as e:
    LOGGER.error(f"Issue with create_manifest method: {e}")

def display_test_results(passfail):

  print()
  print()
  print("FITS FILE COMPARISON TEST RESULTS:")
  print()
  for testname in passfail:
    fits_result = "FAIL"
    if (passfail[testname][1]): fits_result = "PASS"
    string = "{0:40}{1}".format(testname, fits_result)
    print(string) # <------- Possibly add zero size FITS output here

  print()
  print()
  print("BINARY FILE COMPARISON TEST RESULTS:")
  print()
  for testname in passfail:
    binary_result = passfail[testname][0]
    string = "{0:40}{1}".format(testname,binary_result)
    print(string) # <------ Add zero file size output here (DiFX/PCAL)
  print()
  print()
  print('For more information please inspect the full log files located within each test directory *testname*/binary_diff.log  *testname*/fits_diff.log') 
  
def setup_logger():
  LOGGER.debug("Starting logger setup...")


def main():

  parser = argparse.ArgumentParser()
  parser.add_argument("-v","--verbose", action="store_true", help="Print additional output information")
  parser.add_argument("--debug", action="store_true", help="Display extra debugging information")
  parser.add_argument("-g","--generateVDIF", action="store_true", help="Generate VDIF data?")
  parser.add_argument("-u","--updatetest", action="store_true", help="Update the default test results") 
  parser.add_argument("-a","--abstol",help="Absolute tolerance for FITS file comparison (default = 1e-6)",default=1e-6)
  parser.add_argument("-r","--reltol",help="Relative tolerance for FITS file comparison (default = 0.0)",default=0.00)
  parser.add_argument("-d","--download",action="store_true", help="Download and run DiFX on real VLBI data?")
  parser.add_argument("-t","--testgpu",action="store_true", help="run DiFX in gpu mode and compare results with benchmark")
  parser.add_argument("-i","--usebenchmarkimfile", action="store_true", help="Use the benchmark .im file from a previous run rather than running difxcalc") 
  parser.add_argument("-c","--cores",help="Comma delimited list of the processing cores used for data simulation and correlation (default = localhost,localhost,localhost,localhost,localhost,localhost,localhost,localhost,localhost,localhost)",default="localhost,localhost,localhost,localhost,localhost,localhost,localhost,localhost,localhost,localhost")
  parser.add_argument("-n","--numthreads",help="Integer number of threads per processing core.  More than one thread will speed runtime but reduce accuracy. (default=1)",default="1")
  parser.add_argument("-f","--filterautocorrs",action="store_true", help="Filter autocorrelations prior to FITS file comparison")
  parser.add_argument("-m", "--manifest", help="File containing experiment execution control parameters (default: manifest.ini)", default="manifest.ini")
  parser.add_argument("-e", "--exp-root", default="experiments", help="Store the location of the directory housing all available experiments")


  # Loading input args
  input_args = parser.parse_args()
  verbose = input_args.verbose
  debug_out = input_args.debug
  generateVDIF = input_args.generateVDIF
  #generateVDIF = generateVDIF.upper()
  updatetest = input_args.updatetest
  reltol = float(input_args.reltol)
  abstol = float(input_args.abstol)
#  download = input_args.download
#  download = download.upper()
  testgpu = input_args.testgpu
  #testgpu = testgpu.upper()
  usebenchmarkimfile = input_args.usebenchmarkimfile
  #usebenchmarkimfile = usebenchmarkimfile.upper()
  cores = input_args.cores
  manifest_file = input_args.manifest
  filterautocorrs = input_args.filterautocorrs
  #filterautocorrs = filterautocorrs.upper()
  numthreads = input_args.numthreads
  root = Path(input_args.exp_root).resolve()

  reset_manifest = False
  # Logging-Level Config
  if debug_out:
    log_level = logging.DEBUG
  elif verbose: 
    log_level = logging.INFO
  else: 
    log_level = logging.WARNING


  # Logging Config
  logging.basicConfig(
    level=log_level, 
    format="%(levelname)s in %(filename)s: %(message)s"
  )

  # Need to have a conditional that catches whenever -u is being used, to preemptively

  # Logger automatically lets through anything above a debug (INFO, WARNING, ERROR, CRITICAL)
  # The levels of importance are heirarchical in this exact order ^

  LOGGER.info("\nStarting DiFXtest.py...\n")
  setup_logger()
  
  if not root.is_dir():
    LOGGER.error(
      f"Test root directory does not exist or is not a directory: "
      f"{root}"
    )


  LOGGER.info(f"Using test root: {root}\n")
  # Pulling attributes from manifest.ini
  manifest_attr = configparser.ConfigParser()
  manifest_attr.read(manifest_file)

  if not manifest_attr:
    LOGGER.error(f"Could not read manifest file {manifest_file}, aborting")
    exit(1)

  if not manifest_attr.has_section("running_experiments"):
    LOGGER.error(f"Manifest file {manifest_file} is missing ['running_experiments'] section, aborting")
    exit(1)
  
  running_experiments = manifest_attr['running_experiments']

  cpu_testnames = []
  gpu_testnames = []
  enabled_tests = []
  disabled_tests = []
  available_tests = []

  for experiment, run in running_experiments.items(): 
    #logger.debug(f"Experiment names: {experiment}, var type: {type(experiment)}")
    # experiment is a string so we can make a conditional based on the last four of the basename...
    try:
      do_run = running_experiments.getboolean(experiment)
    except ValueError as ve:
      LOGGER.error(
        f"Invalid Boolean value for '{experiment}' in "
        f"[running_experiments] section of {manifest_file}."
        f"Use true/false, yes/no, on/off, or 1/0"
      )
    if experiment.endswith(
      (".tmp", "tests.tgz", "testdata", "difx_version_info.txt")
    ):
      continue
    if(experiment.endswith("-gpu")):
      gpu_testnames.append(experiment)
    else: 
      cpu_testnames.append(experiment)
    
    if do_run:
      enabled_tests.append(experiment)
    if not do_run:
      disabled_tests.append(experiment)


  #logger.info(f"Available tests: {available_tests}\n")

  test_name_list = enabled_tests
  
  available_tests = cpu_testnames + gpu_testnames

  if not test_name_list: 
    parser.error(
      f"No enabled experiments were found in {manifest_file}"
    )

  LOGGER.info(
    "Enabled tests: (%d): %s\n",
    len(enabled_tests),
    ", ".join(enabled_tests),
  )

  LOGGER.info(
    "Disabled tests: (%d): %s\n",
    len(disabled_tests),
    ", ".join(disabled_tests),
  )

  LOGGER.debug(
    "All tests in manifest (%d): %s\n",
    len(available_tests),
    ", ".join(available_tests),
  )

  # Number of stations + 2 is the number of coress needed, need head node
  cores_list = cores.split(",")
  numcores = len(cores_list) + 2
  LOGGER.debug(f"Number of cores for mpirun call: {numcores}")

  # Check basic installation/compatability issues
  check_mpifxcor_installed()

 
  # generate vdifsim config file
  generate_vdifsim_config(cores)
 

  # Grab benchmark results and setup files 
  get_results_and_setup_files()
  #quit()



  # list of test names
  # test_name_list = ["usb-2-station","usb-4-band-2-station","usb-10-station","usb-lsb-10-station","usb-complex-vs-real","lsb-2-station","usb-complex-vs-complex","complex-usb-vs-lsb"]
  # gpu_compatable_test_name_list = ["usb-2-station-gpu","usb-4-band-2-station-gpu","usb-10-station-gpu","usb-lsb-10-station-gpu","usb-complex-vs-real-gpu","lsb-2-station-gpu","usb-complex-vs-complex-gpu","complex-usb-vs-lsb-gpu"]
# 
  # # Dictionary with pass/fail status of each test {"test name" : [binary file same (results), FITS file same (true/false)]} 
  # passfail = {"usb-2-station":["",True],"usb-4-band-2-station":["",True],"usb-10-station":["",True],"usb-lsb-10-station":["",True],"usb-complex-vs-real":["",True],"lsb-2-station":["",True],"usb-complex-vs-complex":["",True],"complex-usb-vs-lsb":["",True]}
# 
  # passfail_new = {}
# 
  # #logger.debug(passfail_new)
# 
# # execute: yes
# # test-type: gpu
# 
  # # Add gpu mode tests to list of tests to be run 
  # if (testgpu == "YES"):
  #   test_name_list.extend(gpu_compatable_test_name_list)
  #    
# 
# 
  # if (testgpu == "YES"):
  #   for item in gpu_compatable_test_name_list:
  #     passfail[item] = ["",True]
  #     new_key = item + "-vs-cpu"
  #     passfail[new_key] = ["",True]


# ------------------------------------------------------------
# Create pass/fail entries only for tests enabled in the manifest.
# ------------------------------------------------------------

  passfail = {
      testname: ["", True]
      for testname in test_name_list
  }

  # Add GPU-vs-CPU comparisons only when both corresponding tests run.
  for testname in gpu_testnames:
      cpu_testname = testname.removesuffix("gpu")
      comparison_name = f"{testname}-vs-cpu"

      # Moving on if cpu test counterpart doesn't exist
      if cpu_testname not in available_tests:
        continue

      if testname in enabled_tests and cpu_testname in enabled_tests:
        passfail[comparison_name] = ["", True]
      elif testname in disabled_tests or cpu_testname in disabled_tests:
        passfail[comparison_name] = ["skipped", True]
        LOGGER.info(
          f"Skipping GPU-vs-CPU comparison {comparison_name} because one or more "
          "corresponding tests are disabled."
        )

  # For backwards compatibility
  if testgpu:
    gpu_tests_enabled_by_manifest = [
        testname
        for testname in test_name_list
        if testname.endswith("-gpu")
    ]

    if gpu_tests_enabled_by_manifest:
        LOGGER.warning(
            "GPU tests are enabled in the manifest but --testgpu was not switched to true. "
            "GPU tests will be skipped: %s",
            ", ".join(gpu_tests_enabled_by_manifest),
        )

    #test_name_list = [
    #    testname
    #    for testname in test_name_list
    #    if not testname.endswith("-gpu")
    #]


  # ADD test name list needs to be exteded with the full path for each test name
  # need to update all functions to take full path rather than test name 
#  print()
#  for ii in range(len(test_name_list)):
#    testpath = get_testdir()
#    test_name_list[ii] = testpath + "/" + test_name_list[ii]
#    print(test_name_list[ii])
  #quit()
  # Unpack tests if they haven't been already
  #print(test_name_list)

  unpackage_tests(test_name_list)


  # print benchmark version info
  record_difx_version_info()
  print_version_info()
 
  set_manifest(root, updatetest)

  tests_run = 0

  if (generateVDIF):

    rm_test_data()
    for testname in available_tests:
      if is_testdata_empty(testname):
        check_vdifsim_installed()
        record_path()
        if not testname.endswith(".gpu"):
          vdifsim_hangup = run_vdifsim(testname)
          LOGGER.debug(f"Printing vdifsim_hangup: {vdifsim_hangup}, type: {type(vdifsim_hangup)}")

  # Run DiFX on all compatable tests 
  for testname in test_name_list:
    if(testname.endswith('.gpu')):
      align_chans_ffts(testname)

    rm_output_files(testname)
    rename_oms_v2d(testname)
    generate_v2d(testname)
    #fix_filelist_path(testname) # Think about just adding this to the end of the generate_v2d method
    set_numthreads(testname,numthreads)
    generate_filelist(testname)
        

    # generated v2d file needs to be edited in order to generated complex sampled data
    if testname in (
      "usb-complex-vs-real", 
      "usb-complex-vs-real-gpu"
    ):
      
      LOGGER.info(f"Applying complex format change to {testname}") 
      add_complex_sampling_to_v2d(testname)

    if testname in (
      "usb-complex-vs-complex", 
      "usb-complex-vs-complex-gpu", 
      "complex-usb-vs-lsb-gpu",
      "complex-usb-vs-lsb"
    ):
      LOGGER.info(f"Applying all-complex format change to {testname}")
      add_complex_sampling_to_v2d_all_complex(testname)


    run_vex2difx(testname)
    #quit()
    if usebenchmarkimfile:
      # copy benchmark im file to working directory
      current_directory = get_testdir()
      working_directory = current_directory + "/" + testname + "/"
      results_directory = working_directory + "/benchmark_results/" 
      benchmark_im_file = get_im_file(results_directory) 
      shutil.copy2(benchmark_im_file, working_directory) 

    run_difxcalc(testname)

    tests_run +=1
  
    if (testname[-3:] == "gpu"):  
      run_mpifxcorr_gpumode(testname, numcores)
    else:
      run_mpifxcorr(testname, numcores)

    run_difx2fits(testname,filterautocorrs)

  LOGGER.debug(f"Number of available tests: {len(available_tests)}")
  
  LOGGER.debug(f"Total tests run: {tests_run}")
  

  # compare .im files
  for testname in test_name_list:
    compare_im_files(testname) # <------ Check this function
  print()

  
    
  # Run comparison with the results, gpu comparisons also compare gpu vs. cpu results
  for key in passfail: 
    if key.endswith("-vs-cpu"):
      passfail[key] = compare_results_gpu_v_cpu(key,abstol,reltol,filterautocorrs)
    else:
      passfail[key] = compare_results(key,abstol,reltol,filterautocorrs)
    
  LOGGER.debug(f"Printing passfail: {passfail}")
  
  # Always want to make sure the gpu test gets added to the tar ball even if it wasn't run
  # this way the setup files and directory will be there if it is run in the future

  if (updatetest): # Do a rework of the repackage tests function, then do a rework of the --update-tests flag... 
    record_difx_version_info()
    set_manifest(root, reset=True)

    for testname in test_name_list:
      update_test(testname, filterautocorrs)
    repackage_tests(available_tests)

  #print(passfail)

  display_test_results(passfail)

if __name__ == "__main__":
    main()



