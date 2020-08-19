from setuptools import find_packages, setup

NAME = 'conv_stft'
VERSION = "0.1.2"
REQUIREMENTS = [
    'numpy',
    'scipy',
]

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name=NAME,
      version=VERSION,
      description="A Conv-STFT/iSTFT implement based on Torch",
      long_description=long_description,
      long_description_content_type="text/markdown",
      url="https://github.com/echocatzh/conv-stft",
      author="Shimin Zhang",
      author_email="shmzhang@npu-aslp.org",
      packages=["conv_stft"],
      install_requires=["numpy","scipy"],
      python_requires=">=3.5",
      license="MIT")