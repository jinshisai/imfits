from setuptools import setup, find_packages

setup(
	name="imfits",
	version='2.3.4',
	description='Read and handle fits files for astronomy easily.',
	author='Jinshi Sai',
	author_email='jn.insa.sai@gmail.com',
	url='https://github.com/jinshisai/imfits',
	packages=find_packages(),
	classifiers=[
	"Programming Language :: Python :: 3",
	"License :: OSI Approved :: MIT License",
	"Operating System :: OS Independent",
    ],
    install_requires=open('requirements.txt').read().splitlines()
)