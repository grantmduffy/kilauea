from setuptools import setup, find_packages

setup(
    name='kilauea',
    version='0.1.0',
    author='Grant Duffy',
    author_email='grantmduffy@gmail.com',
    description='The pythonic vulkan render engine.',
    packages=find_packages(),
    install_requires=[
        'vulkan',
        'numpy',
        'glfw',
        'PyGLM',
        'sphinx'
    ],
)
