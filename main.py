import copy
import os
import re
import dearpygui.dearpygui as dpg
import webbrowser
import yaml

# ML-Agents Variables
mlAgentsData = {}
defaultMlAgentsData = {}

# Hyper Parameter Tuner Variables
hyperParameterTuningData = {}
defaultHyperParameterTuningData = {}
allHyperParameterTuningConfigFiles = []

# Global Width, Global Height
GLOBAL_WIDTH = 800
GLOBAL_HEIGHT = 600
GLOBAL_FONT_SIZE = 1.15

def loadMlAgentsConfig(mlAgentsConfigFile):
    with open(mlAgentsConfigFile, 'r') as f:
        config = yaml.safe_load(f)

    return config

def loadHyperParameterConfig(hyperParameterConfigFile):
    with open(hyperParameterConfigFile, 'r') as f:
        config = yaml.safe_load(f)

    return config

def _help(message):
    last_item = dpg.last_item()
    group = dpg.add_group(horizontal=True)
    dpg.move_item(last_item, parent=group)
    dpg.capture_next_item(lambda s: dpg.move_item(s, parent=group))
    t = dpg.add_text("(?)", color=[0, 255, 0])
    with dpg.tooltip(t):
        dpg.add_text(message)

def _hyperlink(text, address):
    b = dpg.add_button(label=text, callback=lambda:webbrowser.open(address))
    dpg.bind_item_theme(b, "__demo_hyperlinkTheme")

def restore_config(sender, app_data, user_data):
    print("configuration restored") # debugging log

    hyperParameterTuningData["mlagents"]["env_settings"]["env_path"] = defaultHyperParameterTuningData["mlagents"]["env_settings"]["env_path"]
    hyperParameterTuningData["realm_ai"]["behavior_name"] = defaultHyperParameterTuningData["realm_ai"]["behavior_name"]
    hyperParameterTuningData["mlagents"]["default_settings"]["max_steps"] = defaultHyperParameterTuningData["mlagents"]["default_settings"]["max_steps"]

    dpg.set_value(user_data[0], hyperParameterTuningData["mlagents"]["env_settings"]["env_path"])
    dpg.set_value(user_data[1], hyperParameterTuningData["realm_ai"]["behavior_name"])
    dpg.set_value(user_data[2], hyperParameterTuningData["mlagents"]["default_settings"]["max_steps"])

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
    hyperParameterTuningData["mlagents"]["env_settings"]["env_path"] = dpg.get_value(user_data[0])
    hyperParameterTuningData["realm_ai"]["behavior_name"] = dpg.get_value(user_data[1])
    hyperParameterTuningData["mlagents"]["default_settings"]["max_steps"] = int(dpg.get_value(user_data[2]))

    if len(dpg.get_value(user_data[3])) != 0 and dpg.get_value(user_data[3]) != ".yaml":
        newConfigFile = "configs/" + dpg.get_value(user_data[3])
    else:
        newConfigFile = "configs/config.yaml"

    with open(newConfigFile, 'w') as outfile:
        yaml.dump(hyperParameterTuningData, outfile, sort_keys=False)

    print("new configuration created") # debugging log

def prompt_show_config(sender, app_data, user_data):
    # Get all possible config files in current directory to show up
    allHyperParameterTuningConfigFiles = [] # reset list
    directory = "configs/"
    try:
        for file in os.listdir(directory):
            if file.endswith(".yaml"):
                allHyperParameterTuningConfigFiles.append(file)

        allHyperParameterTuningConfigFiles.sort()
    except OSError as e:
        print("Error: %s : %s" % (directory, e.strerror))
        allHyperParameterTuningConfigFiles = []

    # Prompt To Appear - user_data[0] = prompt dropdown component which dynamically changes it list contents
    dpg.configure_item(user_data[0], items=allHyperParameterTuningConfigFiles)
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

    # Themes
    with dpg.theme(tag="__demo_hyperlinkTheme"):
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button, [0, 0, 0, 0])
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, [0, 0, 0, 0])
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, [29, 151, 236, 25])
            dpg.add_theme_color(dpg.mvThemeCol_Text, [29, 151, 236])

    # Tuner Prompt
    with dpg.window(label="Create Configuration and Start Training", modal=True, pos=[GLOBAL_WIDTH/6, GLOBAL_HEIGHT/3] ,id="prompt", show=False):
        dpg.add_text("This will create a new config file and start the training.\nChoose a configuration file:")
        dpg.add_spacer(height=10)
        config_file_to_run = dpg.add_combo(label="config_file", items=allHyperParameterTuningConfigFiles)
        dpg.add_spacer(height=10)
        dpg.add_separator()
        dpg.add_spacer(height=1)

        with dpg.group(horizontal=True):
            dpg.add_button(label="Proceed", width=75, callback=run_training, user_data=[config_file_to_run])
            dpg.add_button(label="Cancel", width=75, callback=lambda: dpg.configure_item("prompt", show=False))

    # Tuner Main Window
    with dpg.window(label="Training Manager", width=GLOBAL_WIDTH-15, height=GLOBAL_HEIGHT-50, no_collapse=True, no_close=True):        
         
        with dpg.collapsing_header(label="ML Agents Training", default_open = True):
            dpg.add_text("Edit the values, press save, and then start training!", color=[232,163,33])
            with dpg.group(horizontal=True):
                dpg.add_text("For more information about the training configuration files", color=[232,163,33])
                _hyperlink("click here", "https://github.com/Unity-Technologies/ml-agents/blob/main/docs/Training-Configuration-File.md")
            dpg.add_spacer(height=10)
            dpg.add_separator()

            dpg.add_text("Hyperparameters", color=[137,207,240])
            dpg.add_input_int(label="batch_size", default_value=int(mlAgentsData["hyperparameters"]["batch_size"]), step=128, min_value=1, min_clamped=True)
            _help("Typical range (Continuous PPO): 512 - 5120")
            dpg.add_input_int(label="buffer_size", default_value=int(mlAgentsData["hyperparameters"]["buffer_size"]), step=128, min_value=32, min_clamped=True)
            _help("ypical range (PPO): 2048 - 409600")
            dpg.add_input_float(label="learning_rate", default_value=float(mlAgentsData["hyperparameters"]["learning_rate"]), format="%e", min_value=1e-9, min_clamped=True, max_value=0.99999, max_clamped=True)
            _help("Typical range: 1e-5 - 1e-3")
            dpg.add_input_float(label="beta", default_value=float(mlAgentsData["hyperparameters"]["beta"]), format="%e", min_value=1e-9, min_clamped=True, max_value=0.49999, max_clamped=True)
            _help("Typical range: 1e-4 - 1e-2")
            dpg.add_input_float(label="epsilon", default_value=float(mlAgentsData["hyperparameters"]["epsilon"]), min_value=1e-9, min_clamped=True, max_value=0.99999, max_clamped=True)
            _help("Typical range: 0.1 - 0.3")
            dpg.add_input_float(label="lambda", default_value=float(mlAgentsData["hyperparameters"]["lambd"]), min_value=1e-9, min_clamped=True, max_value=0.99999, max_clamped=True)
            _help("Typical range: 0.9 - 0.95")
            dpg.add_slider_int(label="num_epoch", default_value=float(mlAgentsData["hyperparameters"]["num_epoch"]), min_value=1, max_value=15)
            _help("Typical range: 3 - 10")
            lr_schedule = ["linear", "constant"]
            dpg.add_combo(label="learning_rate_schedule", items=lr_schedule, default_value=lr_schedule[0])
            _help("linear decays the learning_rate linearly, constant keeps the learning_rate constant")
            
            dpg.add_spacer(height=3)
            dpg.add_text("Network Settings", color=[137,207,240])
            dpg.add_checkbox(label="normalize", default_value=bool(mlAgentsData["network_settings"]["normalize"]))
            _help("Whether normalization is applied to the vector observation inputs")
            hidden_unit_values = [64, 128, 256, 512, 1024]
            dpg.add_combo(label="hidden_units", items=hidden_unit_values, default_value=hidden_unit_values[0])
            _help("Typical range: 32 - 512")
            dpg.add_slider_int(label="num_layers", default_value=int(mlAgentsData["network_settings"]["num_layers"]), min_value=1, max_value=3)
            _help("Typical range: 1 - 3")

            dpg.add_spacer(height=3)
            dpg.add_text("Reward Signals")
            dpg.add_input_float(label="gamma", default_value=float(mlAgentsData["reward_signals"]["extrinsic"]["gamma"]), min_value=1e-9, min_clamped=True, max_value=0.99999, max_clamped=True)
            _help("Typical range: 0.8 - 0.995")
            dpg.add_input_float(label="strength", default_value=float(mlAgentsData["reward_signals"]["extrinsic"]["strength"]), min_value=1e-9, min_clamped=True, max_value=0.99999, max_clamped=True)
            _help("Factor by which to multiply the reward given by the environment.")

            dpg.add_spacer(height=3)
            dpg.add_text("Other Settings", color=[137,207,240])
            dpg.add_slider_int(label="keep_checkpoints", default_value=int(mlAgentsData["keep_checkpoints"]), min_value=1, max_value=10)
            _help("The maximum number of model checkpoints to keep")
            dpg.add_input_int(label="max_steps", default_value=float(mlAgentsData["max_steps"]), step=128, min_value=1, min_clamped=True)
            _help("Typical range: 5e5 - 1e7")
            dpg.add_input_int(label="time_horizon", default_value=int(mlAgentsData["time_horizon"]), step=128, min_value=1, min_clamped=True)
            _help("Typical range: 32 - 2048")
            dpg.add_input_int(label="summary_freq", default_value=int(mlAgentsData["summary_freq"]), step=128, min_value=1, min_clamped=True)
            _help("Number of experiences that needs to be collected before generating/displaying training stats")
            dpg.add_checkbox(label="threaded", default_value=bool(mlAgentsData["threaded"]))
            _help("Allow environments to step while updating the model. This might result in a training speedup, especially when using SAC. For best performance, set to false when using self-play")
            dpg.add_spacer(height=20)

            with dpg.group(horizontal=True):
                dpg.add_button(label="Restore Defaults", callback=restore_config, small=True)
                dpg.add_spacer(width=8)
                dpg.add_button(label="Save Configuration", callback=edit_and_create_config, small=True)
                dpg.add_spacer(width=8)
                dpg.add_button(label="Start Training", callback=prompt_show_config, small=True)
                dpg.add_spacer(height=30)

        with dpg.collapsing_header(label="Hyper Parameter Tuning Configuration"):
            with dpg.tab_bar(label="Tab bar"):
                # Basic Tab
                with dpg.tab(label="Basic"):
                    dpg.add_text("Basic Hyper Parameter Configuration Values:\nEdit the values, press save, and then start training!", color=[232,163,33])
                    dpg.add_spacer(height=10)

                    config_file_name = dpg.add_input_text(label="config_file_name", default_value=".yaml", width=400, hint="new config file name")
                    env_path = dpg.add_input_text(label="env_path", default_value=str(hyperParameterTuningData["mlagents"]["env_settings"]["env_path"]), width=400, hint="env_path of ml-agents")
                    behavior_name = dpg.add_input_text(label="behavior_name", default_value=str(hyperParameterTuningData["realm_ai"]["behavior_name"]), width=400, hint="unity game project name")
                    max_steps = dpg.add_input_int(label="max_steps", default_value=int(hyperParameterTuningData["mlagents"]["default_settings"]["max_steps"]), step=1000, min_value=1, max_value=1e9)
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
    # ML Agents Data
    mlAgentsConfigFile = "ml-agents/ppo.yaml"
    mlAgentsData = loadMlAgentsConfig(mlAgentsConfigFile)["default_settings"]
    # print(mlAgentsData)
    defaultMlAgentsData = copy.deepcopy(mlAgentsData)

    # Hyper Parameter Data
    hyperParameterConfigFile = "configs/bayes.yaml"
    hyperParameterTuningData = loadHyperParameterConfig(hyperParameterConfigFile)
    defaultHyperParameterTuningData = copy.deepcopy(hyperParameterTuningData)

    startGUI()