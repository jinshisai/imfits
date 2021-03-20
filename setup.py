from setuptools import setup
from setuptools import find_packages

setup(
	name="jinshisai",
	version='1.0',
	description='Read and handle fits files for astronomy easily.',
	author='Jinshi Sai',
	author_email='jn.insa.sai@gmail.com',
	url='https://github.com/jinshisai/Imfits',
	packages=find_packages(),
	classifiers=[
	"Programming Language :: Python :: 3",
	"License :: OSI Approved :: MIT License",
	"Operating System :: OS Independent",
    ],
    install_requires=open('requirements.txt').read().splitlines()
)