from setuptools import setup 
from setuptools import find_packages

#with open("README.md", "r") as fh:
#    long_description = fh.read()

setup(
    name =                  'delearn',
    version =               '0.0.1',
    url =                   "https://github.com/Nelson-iitp/delearn",
    author =                "Nelson.S",
    author_email =          "mail.nelsonsharma@gmail.com",
    description =           '~ D E L E A R N ~',
    packages =              ['delearn'],
    classifiers=            ['License :: OSI Approved :: MIT License'],
    package_dir =           { '' : 'module'},
    install_requires =      [],
    include_package_data =  True
)

