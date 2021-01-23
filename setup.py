from setuptools import setup, find_packages

install_requires = [
    "numpy",
    "jupyter",
    "matplotlib",
    "unittest2"
]

setup(
    name="gradplanner",
    version="0.0.1",
    author="Ferenc Török",
    author_email="ferike.trk@gmail.com",
    packages=find_packages("."),
    licence="LICENCE.txt",
    url="https://github.com/ferenctorok/potential_field_planner",
    install_requires=install_requires,
)
