import os
from setuptools import setup, find_packages
import platform
import torch

# Metadata
NAME = "nexaai-gpu"
VERSION = "0.0.3"
DESCRIPTION = "Nexa AI SDK GPU version"
LONG_DESCRIPTION = open(os.path.join(os.path.dirname(__file__), "README.md")).read()
AUTHOR = "Nexa AI"
AUTHOR_EMAIL = "octopus@nexa4ai.com"
URL = "https://github.com/NexaAI/nexa-sdk"

# Package data
package_data = {
    "nexaai": []
}

# Use the TARGET_PLATFORM environment variable to determine the target platform or use the current platform
target_platform = os.environ.get("TARGET_PLATFORM", platform.system())
if target_platform in ['Linux', 'Windows']:
    if torch.cuda.is_available():
        llama_cpp_cmake_args = "-DGGML_CUDA=on"
        stable_diffusion_cpp_cmake_args = '-DSD_CUBLAS=ON'
    elif torch.backends.mps.is_available(): #ROCM
        llama_cpp_cmake_args = "-DGGML_HIPBLAS=on"
        stable_diffusion_cpp_cmake_args = ''    # TODO: refer to 
elif target_platform == 'Darwin':
    if torch.backends.mps.is_available():
        llama_cpp_cmake_args = "-DGGML_METAL=on"
        stable_diffusion_cpp_cmake_args = '-DSD_METAL=ON'

# Read requirements from files
def read_requirements(filename):
    req_file_path = os.path.join(os.path.dirname(__file__), filename)
    with open(req_file_path, 'r') as file:
        return [line.strip() for line in file if line.strip() and not line.startswith('#')]

requirements = read_requirements('requirements.txt')

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    license='MIT',
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    url=URL,
    packages=find_packages(),   # todo
    install_requires=requirements,
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    package_data=package_data,
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "nexa-cli=nexa.cli.entry:main",
            "nexa=nexa.cli.entry:main",
            "nexaai=nexa.cli.entry:main",
            "nexai=nexa.cli.entry:main",
        ],
    },
    cmake_args = [
        llama_cpp_cmake_args,
        stable_diffusion_cpp_cmake_args
    ],
    setup_requires=[
        "scikit-build-core[pyproject]>=0.9.2",
    ],
    package_data={
        '': ['dependency/llama.cpp/*', 'dependency/stable-diffusion.cpp/*'],
    },
    zip_safe=False,    
)
"""
# test

rm -rf dist build nexaai.egg-info
python -m build
pip install dist/*.whl --force-reinstall # to install gguf
pip install 'nexaai[onnx]' --find-links=dist # to install gguf and onnx

# upload

twine upload dist/*.whl
"""