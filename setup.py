from setuptools import setup, find_packages

setup(
    name='realm_gui',
    version='0.0.1-beta',
    description="REALM_AI's Training Manager",
    # url='#',
    author='REALM_AI',
    install_requires=[
        'dearpygui >= 1.0.2',
        'PyYAML >= 6.0',
        "mlagents",
        "realm_ai" # to-be-added 
    ],
    python_requires=">=3.7", # https://github.com/hoffstadt/DearPyGui/issues/1258
    packages=find_packages(include=['realm_gui']),
    package_data={'realm_gui': ['hyperparameter_configs/*.yaml', 'ml_agents_configs/*.yaml']},
    entry_points={
        "console_scripts": [
            "realm-gui=realm_gui.main:main",
        ]
    }
)