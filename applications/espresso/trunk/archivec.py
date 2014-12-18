#!/usr/bin/python
# archive DiFX data files to a remote machine. Tar the plethora of small files
# before transfer. Transfer larger files unmodified. Preserve the original
# directory structure. Uses globus-url-copy by default or scp/ssh on request. 
# Cormac Reynolds: Jan 2012

import optparse, os, re, subprocess, sys, shutil

def taritup(tardir, tarfile, infile, gzip=False):

    taroptions = options.taroptions
    if gzip:
        taroptions = " ".join([taroptions, '-z'])
    command = " ".join(['tar', taroptions, '-cf', archdir+tarfile, infile])
    if options.verbose:
        print '\n' + command
    subprocess.check_call(command, shell=True, stdout=sys.stdout, stderr=subprocess.PIPE)

    # and print tar listing for reference
    command = " ".join(["tar -tf", tardir+tarfile, ">", tardir+tarfile+'.list'])
    subprocess.check_call(command, shell=True, stdout=sys.stdout, stderr=subprocess.PIPE)

usage = '''%prog <path> <destination>
will transfer <path> and all its subdirectories to <destination> on data.ivec.org using ashell.py. Most files are tarred before transfer, but special files and large files are transferred unmodified. Files are first tarred/copied to /data/corr/Archive/ before transfer.

e.g.
%prog /data/corr/corrdat/vt13b VLBI/Archive/Curtin/vt13/vt13b
'''

parser = optparse.OptionParser(usage=usage, version='%prog ' + '1.0')
parser.add_option( "--maxtarsize", "-m",
        type='float', dest="maxtarsize", default=1000,
        help='files larger than MAXTARSIZE (MB) will be transferred untarred [default = %default]' )
parser.add_option( "--taroptions", "-t",
        type='str', dest="taroptions", default=' ',
        help='specify additional options to pass to tar' )
parser.add_option( "--verbose", "-v",
        dest="verbose", action="store_true", default=False,
        help='Be verbose' )

(options, args) = parser.parse_args()

if len(args) < 2:
    parser.print_help()
    parser.error("Give source and destination directories!")


# get the list of subdirectories. Note that we don't want to tar large files,
# or files that archive users want to access directly.
# In each  subdirectory form a list of files to be tarred, and a list of files
# to transfer unmodified (both tarred files and large files will be
# transferred).

#archdir = args[1] 
expname = os.path.normpath(args[0]).split('/')[-1]
archdir = '/data/corr/Archive/' + expname + os.sep
mark4file = str()
os.chdir(args[0])
tarlist = str()
transfer = []
for filename in os.listdir(os.curdir):

    # deal with Mark4 output, clocks and test as special cases
    if re.search('^\d\d\d\d$', filename):
        mark4file = filename
        continue
    if filename == 'clocks':
        continue
    if filename == 'test':
        continue

    # certain file names never get tarred 
    notar_ext = ['.fits', '.mark4', '.tar']
    fileWithPath = os.path.join(os.path.abspath(os.curdir), filename)
    notar = False
    for extension in notar_ext:
        if re.search(extension, filename, re.IGNORECASE):
            notar = True
            break

    # only tar small files
    if os.path.getsize(fileWithPath)/1e6 > options.maxtarsize:
        notar = True

    if (os.path.exists(fileWithPath) and notar):
        # transfer this large file without tarring
        transfer.append(re.escape(fileWithPath))
    else:
        # add to list of files to be tarred
        tarlist += ' ' + re.escape(filename)

tarfile =  os.path.basename(os.path.abspath(os.curdir)) + '.tar'
#tardir = archdir + os.sep 

# create the output directory
command = " ".join(['mkdir -p', archdir ])
subprocess.check_call(command, shell=True, stdout=sys.stdout)


# tar up small files in this directory to Archive area
if tarlist:
    taritup(archdir, tarfile, tarlist)

# transfer each of the large files in turn
for srcfile in transfer:
    command = " ".join(["cp -l", srcfile, archdir])
    if options.verbose:
        print '\n' + command
    subprocess.check_call(command, shell=True, stdout=sys.stdout)

# now tar up the clocks subdirectory
taritup(archdir, 'clocks.tar', 'clocks')

# and the mark4 output dir
if mark4file:
    taritup(archdir, expname.upper()+'.MARK4.tar.gz', mark4file, gzip=True)


# now archive the lot to data.ivec.org
os.chdir(archdir)
#command = " ".join(["ashell.py login + delegate 100"])
#subprocess.check_call(command, shell=True, stdout=sys.stdout, stderr=sys.stderr)
while True:
    try:
        command = " ".join(['ashell.py "cf', args[1], "+ put", archdir, '"'])
        subprocess.check_call(command, shell=True, stdout=sys.stdout, stderr=sys.stderr)
        break
    except KeyboardInterrupt:
        raise Exception('Forced quit')
    else:
        print 'trying again'
        #command = " ".join(['ashell.py "login + delegate 100"'])
        #subprocess.check_call(command, shell=True, stdout=sys.stdout, stderr=sys.stderr)

shutil.rmtree(archdir)

print 'All done!'
