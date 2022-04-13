# Import the os module, for the os.walk function
import os
import csv


rootDir = '/usr/local/'
for dirName, subdirList, fileList in os.walk(rootDir):
    if(os.path.isdir("/usr/local/cuda")):
        with open("envset.sh", "w") as fp:
            string1 = "export PATH=/usr/local/cuda/bin:$PATH\nexport LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH\nexport CUDA_PATH=\"/usr/local/cuda\""
            fp.write(string1)
    else:
        with open("envset.sh", "w") as fp:
            string1 = "export PATH=/usr/local/cuda/bin:$PATH\nexport LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH\nexport CUDA_PATH=\"/usr/local/cuda\""
            fp.write(string1)
