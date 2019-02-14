'''
This script that will run all analyses,
producing all figures.

The figures will not be displayed until all scripts have run.
'''



import os
from IPython import get_ipython
from matplotlib import pyplot



#assemble script names:
fnames   = []
fnames.append('fig01_nonuniform.py')
fnames.append('fig02_terminology.py')
fnames.append('fig03_terminology2D.py')
fnames.append('fig04_cluster.py')
fnames.append('fig05_models.py')
fnames.append('fig06_models_hetero.py')
fnames.append('fig07_sfs.py')
fnames.append('fig08_sfs_hetero.py')
fnames.append('fig09_fpdistr.py')
fnames.append('fig10_clusterdistr.py')



#run scripts:
pyplot.close('all')
ipython  = get_ipython()
dir0     = os.path.dirname( __file__ )
for fname in fnames:
	print('Running script: %s ...' %fname)
	fnamePY = os.path.join(dir0, fname)
	ipython.magic('run %s' %fnamePY)
	