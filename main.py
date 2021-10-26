import dearpygui.dearpygui as dpg
import yaml
import copy

# Global original config data
configData = {}
defaultConfigData = {}

# Global Width, Global Height
GLOBAL_WIDTH = 700
GLOBAL_HEIGHT = 400

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
    
    ** we can keep adding to this list
"""
def edit_config(sender, app_data, user_data):
    print("configuration saved") # debugging log
    # config = user_data[0]

    # Overwrite existing config file
    configData["mlagents"]["env_settings"]["env_path"] = dpg.get_value(user_data[0])
    configData["realm_ai"]["behavior_name"] = dpg.get_value(user_data[1])
    configData["mlagents"]["default_settings"]["max_steps"] = int(dpg.get_value(user_data[2]))


"""
    user_data[0] = new file name

    ** we can keep adding to this list
"""
def create_config(sender, app_data, user_data):
    newConfigFile = user_data[0]

    with open(newConfigFile, 'w') as outfile:
        yaml.dump(configData, outfile, sort_keys=False)

    print("new configuration created") # debugging log
    dpg.configure_item("prompt", show=False)


"""
    user_data[0] = new file name

    ** we can keep adding to this list
"""
def prompt_config(sender, app_data, user_data):
    # Prompt After You Press Save
    with dpg.window(label="Create Configuration and Start Training", modal=True, pos=[GLOBAL_WIDTH/6, GLOBAL_HEIGHT/3] ,id="prompt"):
        dpg.add_text("This will create a new config file and start the training.\nAre you sure your configuration is set correctly?")
        dpg.add_separator()
        
        with dpg.group(horizontal=True):
            dpg.add_button(label="Proceed", width=75, callback=create_config, user_data=[user_data[0]])
            dpg.add_button(label="Cancel", width=75, callback=lambda: dpg.configure_item("prompt", show=False))

    
def startGUI(newConfigFile):
    dpg.create_context()
    dpg.create_viewport(title="REALM-AI", width=GLOBAL_WIDTH, height=GLOBAL_HEIGHT)
    dpg.set_global_font_scale(1.1)

    # Main Window
    with dpg.window(label="Hyper Parameter Tuning Configuration", width=GLOBAL_WIDTH, height=GLOBAL_HEIGHT, no_collapse=True, no_close=True):        
        with dpg.tab_bar(label="Tab bar"):
            with dpg.tab(label="Basic"):
                dpg.add_text("Basic Hyper Parameter Configuration Values:\nEdit the values, press save, and then start training!", color=[232,163,33])
                dpg.add_spacer(height=10)

                env_path = dpg.add_input_text(label="env_path", default_value=str(configData["mlagents"]["env_settings"]["env_path"]), width=300, hint="env_path of ml-agents")
                behavior_name = dpg.add_input_text(label="behavior_name", default_value=str(configData["realm_ai"]["behavior_name"]), width=300, hint="unity game project name")
                max_steps = dpg.add_input_int(label="max_steps", default_value=int(configData["mlagents"]["default_settings"]["max_steps"]), step=1000, min_value=1, max_value=1e9)
                dpg.add_spacer(height=25)

                with dpg.group(horizontal=True):
                    dpg.add_button(label="Restore Defaults", callback=restore_config, user_data=[env_path, behavior_name, max_steps], small=True)
                    dpg.add_spacer(width=8)
                    dpg.add_button(label="Save Configuration", callback=edit_config, user_data=[env_path, behavior_name, max_steps], small=True)
                    dpg.add_spacer(width=8)
                    dpg.add_button(label="Start Training", callback=prompt_config, user_data=[newConfigFile], small=True)

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
    configFile = "bayes.yaml"
    newConfigFile = "bayesNew.yaml"
    configData = loadConfig(configFile)
    defaultConfigData = copy.deepcopy(configData)

    startGUI(newConfigFile)