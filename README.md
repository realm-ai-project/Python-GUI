# Python-GUI

Library and Documentation here: 
1) https://github.com/hoffstadt/DearPyGui
2) https://dearpygui.readthedocs.io/en/latest/

Install first by doing `pip3 install -r requirements.txt` and then run `main.py`

Note: Does not work on Python 3.6, see https://github.com/hoffstadt/DearPyGui/issues/1258

## Functionality
1) Edit fields as you like
2) `Save Configuration` button -> creates config file under `configs/<filename>`. It overwrites if the file already exists.
3) `Restore Defaults` button -> restore defaults based on original master configuration file: `configs/bayes.yaml`.
4) `Start Training` button -> pick which config file you want and run the hyper parameter tuning code.