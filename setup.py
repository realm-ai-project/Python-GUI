from setuptools import setup, find_packages

setup(
    name='realm_gui',
    version='0.0.1',
    description="REALM_AI's Training Manager",
    # url='#',
    author='REALM_AI',
    install_requires=['dearpygui >= 1.0.2', 'PyYAML >= 6.0'],
    packages=find_packages(include=['realm_gui']),
    entry_points={
        "console_scripts": [
            "realm-gui=realm_tune.main:main",
        ]
    }
)