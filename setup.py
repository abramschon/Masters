from setuptools import setup

setup(
    name="MaxEnt", # name on PyPI not necessarily the same as the name of the code
    version="0.0.1", # 0.0.x numbers imply unstable versions
    description='Offers a range of maximum entropy models', # what your package does
    py_modules=["MaxEnt"], # list of the python code modules that you install 
    package_dir={'':'src'} # specifies code is in src directory
)