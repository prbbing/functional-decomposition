import itertools, json, os
import numpy           as np

from  numpy.lib.format import open_memmap
from  Tools.PrintMgr   import *

# Format path elements from pathTpl, create the directory chain,
#  return the full path, filename and extension.
def _mkPath(pathTpl, *args, **kwargs):
    path     = [x.format(*args, **kwargs) for x in pathTpl]
    FullPath = os.path.join(*path)
    Ext      = os.path.splitext(path[-1])[1].lower()

    for n in range(1, len(path)):
        try:
            os.mkdir( os.path.join(*path[:n]) )
        except OSError:
            pass

    return FullPath, path[-1], Ext

# Atomic file creation method
def _acreate(filename):
    fd = os.open(filename, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0644)
    return os.fdopen(fd, 'wb')

def _fcreate(filename):
    fd = os.open(filename, os.O_CREAT |             os.O_WRONLY, 0644)
    return os.fdopen(fd, 'wb')

# Save a sparse symmetric array.
def _saveSparseSym(Mx, file):
    idx   = np.ndindex(*Mx.shape)
    sdict = { tuple(sorted(i)) : Mx[i] for i in idx if Mx[i] != 0 }

    keys  = zip(*sdict.keys())
    vals  = sdict.values()

    json.dump( (Mx.shape, keys, vals), file)

# Load a spare symmetric array.
def _loadSparseSym(file):
    shp, keys, vals = json.load( file )
    Mx              = np.zeros(shp)

    for x in itertools.permutations(keys):
        Mx[x] = vals

    return Mx

# Cache / calculate a symmetric, multi-dimensional numpy array.
def SymArray(*pathTpl, **warg):
    def owrap(func):
        def wrap(self, *args, **kwargs):
            FullPath, Name, Ext = _mkPath(pathTpl, *args, self=self, **kwargs)
            desc                = kwargs.get("desc", "")

            try:
                with open(FullPath, 'rb') as file:
                    pini("Loading %s (%s)" % (desc, Name) )
                    M = _loadSparseSym(file)
                    pend("Success")
                    return M
            except IOError:
                pass

            pini("Calculating %s" % desc)
            M = func(self, *args)
            pend("Done")

            with open(FullPath, 'wb') as file:
                pini("Saving %s (%s)" % (desc, Name))
                _saveSparseSym(M, file)
                pend("Success")

            return M
        return wrap
    return owrap

# Cache / calculate a generic numpy array or JSON-serializable object.
def _fsave(file, Ext, res):
    if Ext == ".npy":
        np.save  (file, res)
    elif Ext== ".json":
        json.dump(res, file)
    else:
        print "Unknown file extension '%s' from file '%s'." % (Ext, Name)
        
def _fload(Ext, FullPath):
    if Ext == ".npy":
        return np.load(FullPath, mmap_mode='r')
    elif Ext==".json":
        with open(FullPath, 'rb') as file:
            return json.load(file)
    else:
        print "Unknown file extension '%s' from file '%s'." % (Ext, Name)
        raise ValueError

def Element(*pathTpl):
    def owrap(func):
        def wrap(self, *args, **kwargs):
            FullPath, Name, Ext = _mkPath(pathTpl, *args, self=self, **kwargs)

            # Try to load the file
            try:
                return _fload(Ext, FullPath)
            except (ValueError, IOError):
                pass

            # Otherwise, re-create the file and re-compute contents.
            with _fcreate(FullPath) as file:
                res = func(self, *args, **kwargs)
                _fsave(file, Ext, res)
            return _fload(Ext, FullPath)
        return wrap
    return owrap

def AtomicElement(*pathTpl):
    def owrap(func):
        def wrap(self, *args, **kwargs):
            FullPath, Name, Ext = _mkPath(pathTpl, *args, self=self, **kwargs)

            # Try an atomic file creation; if successful, run and save wrapped calculation.
            try:
                with _acreate(FullPath) as file:
                    res = func(self, *args, **kwargs)
                    _fsave(file, Ext, res)
            except OSError:
                pass

            # Otherwise, the file exists, so just load it.
            try:
                return _fload(Ext, FullPath)
            except (ValueError, IOError):
                return 0.0
        return wrap
    return owrap
