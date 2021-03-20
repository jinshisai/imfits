from setuptools import setup, find_packages

setup(
	name="imfits",
	version='1.1',
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