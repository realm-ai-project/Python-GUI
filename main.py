import copy
import os
import re
import dearpygui.dearpygui as dpg
import yaml

# Global original config data
configData = {}
defaultConfigData = {}
allConfigFiles = []

# Global Width, Global Height
GLOBAL_WIDTH = 700
GLOBAL_HEIGHT = 400
GLOBAL_FONT_SIZE = 1.15

def loadConfig(configFile):
    with open(configFile, 'r') as f:
        config = yaml.safe_load(f)

    return config

def restore_config(sender, app_data, user_data):
    print("configuration restored") # debugging log

    configData["mlagents"]["env_settings"]["env_path"] = defaultConfigData["mlagents"]["env_settings"]["env_path"]
    configData["realm_ai"]["behavior_name"] = defaultConfigData["realm_ai"]["behavior_name"]
    configData["mlagents"]["default_settings"]["max_steps"] = defaultConfigData["mlagents"]["default_settings"]["max_steps"]

    dpg.set_value(user_data[0], configData["mlagents"]["env_settings"]["env_path"])
    dpg.set_value(user_data[1], configData["realm_ai"]["behavior_name"])
    dpg.set_value(user_data[2], configData["mlagents"]["default_settings"]["max_steps"])


# https://dearpygui.readthedocs.io/en/latest/tutorials/item-usage.html?highlight=user_data#callbacks
"""
    user_data[0] = env_path
    user_data[1] = behavior_name
    user_data[2] = max_steps
    user_data[3] = config_file_name

    ** we can keep adding to this list
"""
def edit_and_create_config(sender, app_data, user_data):
    print("configuration saved") # debugging log

    # Overwrite existing config file
    configData["mlagents"]["env_settings"]["env_path"] = dpg.get_value(user_data[0])
    configData["realm_ai"]["behavior_name"] = dpg.get_value(user_data[1])
    configData["mlagents"]["default_settings"]["max_steps"] = int(dpg.get_value(user_data[2]))

    if len(dpg.get_value(user_data[3])) != 0:
        newConfigFile = "configs/" + dpg.get_value(user_data[3])
    else:
        newConfigFile = "configs/config.yaml"

    with open(newConfigFile, 'w') as outfile:
        yaml.dump(configData, outfile, sort_keys=False)

    print("new configuration created") # debugging log

def prompt_show_config(sender, app_data, user_data):
    # Get all possible config files in current directory to show up
    allConfigFiles = [] # reset list
    directory = "configs/"
    try:
        for file in os.listdir(directory):
            if file.endswith(".yaml"):
                allConfigFiles.append(file)

        allConfigFiles.sort()
    except OSError as e:
        print("Error: %s : %s" % (directory, e.strerror))
        allConfigFiles = []

    # Prompt To Appear - user_data[0] = prompt dropdown component which dynamically changes it list contents
    dpg.configure_item(user_data[0], items=allConfigFiles)
    dpg.configure_item("prompt", show=True)


"""
    user_data[0] = config_file_to_run

    ** we can keep adding to this list
"""
def run_training(sender, app_data, user_data):
    config_file_to_run = dpg.get_value(user_data[0])
    dpg.configure_item("prompt", show=False)

    print("The config file that is being run is: ", config_file_to_run)


def startGUI():
    dpg.create_context()
    dpg.create_viewport(title="REALM-AI", width=GLOBAL_WIDTH, height=GLOBAL_HEIGHT)
    dpg.set_global_font_scale(GLOBAL_FONT_SIZE)

    # Prompt
    with dpg.window(label="Create Configuration and Start Training", modal=True, pos=[GLOBAL_WIDTH/6, GLOBAL_HEIGHT/3] ,id="prompt", show=False):
        dpg.add_text("This will create a new config file and start the training.\nChoose a configuration file:")
        dpg.add_spacer(height=10)
        config_file_to_run = dpg.add_combo(label="config_file", items=allConfigFiles)
        dpg.add_spacer(height=10)
        dpg.add_separator()
        dpg.add_spacer(height=1)

        with dpg.group(horizontal=True):
            dpg.add_button(label="Proceed", width=75, callback=run_training, user_data=[config_file_to_run])
            dpg.add_button(label="Cancel", width=75, callback=lambda: dpg.configure_item("prompt", show=False))

    # Main Window
    with dpg.window(label="Hyper Parameter Tuning Configuration", width=GLOBAL_WIDTH, height=GLOBAL_HEIGHT, no_collapse=True, no_close=True):        
        with dpg.tab_bar(label="Tab bar"):
            # Basic Tab
            with dpg.tab(label="Basic"):
                dpg.add_text("Basic Hyper Parameter Configuration Values:\nEdit the values, press save, and then start training!", color=[232,163,33])
                dpg.add_spacer(height=10)

                config_file_name = dpg.add_input_text(label="config_file_name", default_value=".yaml", width=400, hint="new config file name")
                env_path = dpg.add_input_text(label="env_path", default_value=str(configData["mlagents"]["env_settings"]["env_path"]), width=400, hint="env_path of ml-agents")
                behavior_name = dpg.add_input_text(label="behavior_name", default_value=str(configData["realm_ai"]["behavior_name"]), width=400, hint="unity game project name")
                max_steps = dpg.add_input_int(label="max_steps", default_value=int(configData["mlagents"]["default_settings"]["max_steps"]), step=1000, min_value=1, max_value=1e9)
                dpg.add_spacer(height=25)

                with dpg.group(horizontal=True):
                    dpg.add_button(label="Restore Defaults", callback=restore_config, user_data=[env_path, behavior_name, max_steps], small=True)
                    dpg.add_spacer(width=8)
                    dpg.add_button(label="Save Configuration", callback=edit_and_create_config, user_data=[env_path, behavior_name, max_steps, config_file_name], small=True)
                    dpg.add_spacer(width=8)
                    dpg.add_button(label="Start Training", callback=prompt_show_config, user_data=[config_file_to_run], small=True)

            # Advanced Tab
            with dpg.tab(label="Advanced"):
                dpg.add_text("Advanced Hyper Parameter Configuration Values:\nDo not edit if you are not familar with RL. Remember to save!", color=[232,163,33])
                dpg.add_spacer(height=10)
                alg = ["bayes", "gridsearch"]
                algorithm = dpg.add_combo(label="algorithm", items=alg, default_value=alg[0])

    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.start_dearpygui()
    dpg.destroy_context()

if __name__ == '__main__':
    configFile = "configs/bayes.yaml"
    configData = loadConfig(configFile)
    defaultConfigData = copy.deepcopy(configData)

    startGUI()