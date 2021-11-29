import argparse
import copy
import os
import re
import runner
import webbrowser
import dearpygui.dearpygui as dpg
import yaml

# Display Variables
showMlAgents = False
showHyperParameter = False

# ML-Agents Variables
mlAgentsData = {}
defaultMlAgentsData = {}
allMlAgentsConfigFiles = []

# Hyper Parameter Tuner Variables
hyperParameterTuningData = {}
defaultHyperParameterTuningData = {}
allHyperParameterTuningConfigFiles = []

# Global Window Variables
GLOBAL_WIDTH = 1000
GLOBAL_HEIGHT = 800
GLOBAL_FONT_SIZE = 1.15

# Master Configuration Files
MASTER_MLAGENTS_FILE = "ppo.yaml"
MASTER_HYPERPARAMETERS_FILE = "bayes.yaml"

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

def restore_ml_config(sender, app_data, user_data):
    print("ml configuration restored") # debugging log

    mlAgentsData["hyperparameters"]["batch_size"] = int(defaultMlAgentsData["hyperparameters"]["batch_size"])
    mlAgentsData["hyperparameters"]["buffer_size"] = int(defaultMlAgentsData["hyperparameters"]["buffer_size"])
    mlAgentsData["hyperparameters"]["learning_rate"] = float(defaultMlAgentsData["hyperparameters"]["learning_rate"])
    mlAgentsData["hyperparameters"]["beta"] = float(defaultMlAgentsData["hyperparameters"]["beta"])
    mlAgentsData["hyperparameters"]["epsilon"] = float(defaultMlAgentsData["hyperparameters"]["epsilon"])
    mlAgentsData["hyperparameters"]["lambd"] = float(defaultMlAgentsData["hyperparameters"]["lambd"])
    mlAgentsData["hyperparameters"]["num_epoch"] = float(defaultMlAgentsData["hyperparameters"]["num_epoch"])
    mlAgentsData["hyperparameters"]["learning_rate_schedule"] = defaultMlAgentsData["hyperparameters"]["learning_rate_schedule"]
    
    mlAgentsData["network_settings"]["normalize"] = defaultMlAgentsData["network_settings"]["normalize"]
    mlAgentsData["network_settings"]["hidden_units"] = int(defaultMlAgentsData["network_settings"]["hidden_units"])
    mlAgentsData["network_settings"]["num_layers"] = int(defaultMlAgentsData["network_settings"]["num_layers"])

    mlAgentsData["reward_signals"]["extrinsic"]["gamma"] = float(defaultMlAgentsData["reward_signals"]["extrinsic"]["gamma"])
    mlAgentsData["reward_signals"]["extrinsic"]["strength"] = float(defaultMlAgentsData["reward_signals"]["extrinsic"]["strength"])

    mlAgentsData["keep_checkpoints"] = int(defaultMlAgentsData["keep_checkpoints"])
    mlAgentsData["max_steps"] = int(float(defaultMlAgentsData["max_steps"]))
    mlAgentsData["time_horizon"] = int(defaultMlAgentsData["time_horizon"])
    mlAgentsData["summary_freq"] = int(defaultMlAgentsData["summary_freq"])
    mlAgentsData["threaded"] = defaultMlAgentsData["threaded"]

    dpg.set_value(user_data[0], mlAgentsData["hyperparameters"]["batch_size"])
    dpg.set_value(user_data[1], mlAgentsData["hyperparameters"]["buffer_size"])
    dpg.set_value(user_data[2], mlAgentsData["hyperparameters"]["learning_rate"])
    dpg.set_value(user_data[3], mlAgentsData["hyperparameters"]["beta"])
    dpg.set_value(user_data[4], mlAgentsData["hyperparameters"]["epsilon"])
    dpg.set_value(user_data[5], mlAgentsData["hyperparameters"]["lambd"])
    dpg.set_value(user_data[6], mlAgentsData["hyperparameters"]["num_epoch"])
    dpg.set_value(user_data[7], mlAgentsData["hyperparameters"]["learning_rate_schedule"])

    dpg.set_value(user_data[8], mlAgentsData["network_settings"]["normalize"])
    dpg.set_value(user_data[9], mlAgentsData["network_settings"]["hidden_units"])
    dpg.set_value(user_data[10], mlAgentsData["network_settings"]["num_layers"])

    dpg.set_value(user_data[11], mlAgentsData["reward_signals"]["extrinsic"]["gamma"])
    dpg.set_value(user_data[12], mlAgentsData["reward_signals"]["extrinsic"]["strength"])

    dpg.set_value(user_data[13], mlAgentsData["keep_checkpoints"])
    dpg.set_value(user_data[14], mlAgentsData["max_steps"])
    dpg.set_value(user_data[15], mlAgentsData["time_horizon"])
    dpg.set_value(user_data[16], mlAgentsData["summary_freq"])
    dpg.set_value(user_data[17], mlAgentsData["threaded"])

def restore_hyperparameter_config(sender, app_data, user_data):
    print("hyperparameter configuration restored") # debugging log

    hyperParameterTuningData["mlagents"]["env_settings"]["env_path"] = defaultHyperParameterTuningData["mlagents"]["env_settings"]["env_path"]
    hyperParameterTuningData["realm_ai"]["behavior_name"] = defaultHyperParameterTuningData["realm_ai"]["behavior_name"]
    hyperParameterTuningData["mlagents"]["default_settings"]["max_steps"] = defaultHyperParameterTuningData["mlagents"]["default_settings"]["max_steps"]

    dpg.set_value(user_data[0], hyperParameterTuningData["mlagents"]["env_settings"]["env_path"])
    dpg.set_value(user_data[1], hyperParameterTuningData["realm_ai"]["behavior_name"])
    dpg.set_value(user_data[2], hyperParameterTuningData["mlagents"]["default_settings"]["max_steps"])

def edit_and_create_mlagents_config(sender, app_data, user_data):
    print("ml-agents configuration saved") # debugging log

    # Overwrite existing ml-agents config file
    mlAgentsData["hyperparameters"]["batch_size"] = dpg.get_value(user_data[0])
    mlAgentsData["hyperparameters"]["buffer_size"] = dpg.get_value(user_data[1])
    mlAgentsData["hyperparameters"]["learning_rate"] = dpg.get_value(user_data[2])
    mlAgentsData["hyperparameters"]["beta"] = dpg.get_value(user_data[3])
    mlAgentsData["hyperparameters"]["epsilon"] = dpg.get_value(user_data[4])
    mlAgentsData["hyperparameters"]["lambd"] = dpg.get_value(user_data[5])
    mlAgentsData["hyperparameters"]["num_epoch"] = dpg.get_value(user_data[6])
    mlAgentsData["hyperparameters"]["learning_rate_schedule"] = dpg.get_value(user_data[7])

    mlAgentsData["network_settings"]["normalize"] = dpg.get_value(user_data[8])
    mlAgentsData["network_settings"]["hidden_units"] = int(dpg.get_value(user_data[9]))
    mlAgentsData["network_settings"]["num_layers"] = dpg.get_value(user_data[10])

    mlAgentsData["reward_signals"]["extrinsic"]["gamma"] = dpg.get_value(user_data[11])
    mlAgentsData["reward_signals"]["extrinsic"]["strength"] = dpg.get_value(user_data[12])

    mlAgentsData["keep_checkpoints"] = dpg.get_value(user_data[13])
    mlAgentsData["max_steps"] = dpg.get_value(user_data[14])
    mlAgentsData["time_horizon"] = dpg.get_value(user_data[15])
    mlAgentsData["summary_freq"] = dpg.get_value(user_data[16])
    mlAgentsData["threaded"] = dpg.get_value(user_data[17])

    reformattedMlAgentsData = {}
    reformattedMlAgentsData["default_settings"] = mlAgentsData

    if len(dpg.get_value(user_data[18])) != 0 and dpg.get_value(user_data[18]) != ".yaml":
        newConfigFile = "ml-agents-configs/" + dpg.get_value(user_data[18])
    else:
        newConfigFile = "ml-agents-configs/config.yaml"

    with open(newConfigFile, 'w') as outfile:
        yaml.dump(reformattedMlAgentsData, outfile, sort_keys=False)

    print("new ml-agents configuration created") # debugging log

def edit_and_create_hyperparameter_config(sender, app_data, user_data):
    print("hyperparameter configuration saved") # debugging log

    # Overwrite existing hyperparameter config file
    hyperParameterTuningData["mlagents"]["env_settings"]["env_path"] = dpg.get_value(user_data[0])
    hyperParameterTuningData["realm_ai"]["behavior_name"] = dpg.get_value(user_data[1])
    hyperParameterTuningData["mlagents"]["default_settings"]["max_steps"] = int(dpg.get_value(user_data[2]))

    if len(dpg.get_value(user_data[3])) != 0 and dpg.get_value(user_data[3]) != ".yaml":
        newConfigFile = "hyperparameter-configs/" + dpg.get_value(user_data[3])
    else:
        newConfigFile = "hyperparameter-configs/config.yaml"

    with open(newConfigFile, 'w') as outfile:
        yaml.dump(hyperParameterTuningData, outfile, sort_keys=False)

    print("new hyperparameter configuration created") # debugging log

def prompt_show_hyperparameter_config(sender, app_data, user_data):
    # Get all possible config files in current directory to show up
    allHyperParameterTuningConfigFiles = [] # reset list
    directory = "hyperparameter-configs/"
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
    dpg.configure_item("hyperparameter_prompt", show=True)

def prompt_show_ml_agents_config(sender, app_data, user_data):
    # Get all possible config files in current directory to show up
    allMlAgentsConfigFiles = [] # reset list
    directory = "ml-agents-configs/"
    try:
        for file in os.listdir(directory):
            if file.endswith(".yaml"):
                allMlAgentsConfigFiles.append(file)

        allMlAgentsConfigFiles.sort()
    except OSError as e:
        print("Error: %s : %s" % (directory, e.strerror))
        allMlAgentsConfigFiles = []

    # Prompt To Appear - user_data[0] = prompt dropdown component which dynamically changes it list contents
    dpg.configure_item(user_data[0], items=allMlAgentsConfigFiles)
    dpg.configure_item("mlagents_prompt", show=True)


# user_data[0] = mlagents_config_file_to_run
def run_training(sender, app_data, user_data):
    mlagents_config_file_to_run = dpg.get_value(user_data[0])
    dpg.configure_item("mlagents_prompt", show=False)

    config_path = "\"" + os.path.abspath(os.getcwd()) + "\\ml-agents-configs\\" + mlagents_config_file_to_run + "\""
    runner.runMLagents(config_path, hyperParameterTuningData["realm_ai"]["behavior_name"])


# user_data[0] = hyperparameter_config_file_to_run
def run_tune_and_training(sender, app_data, user_data):
    hyperparameter_config_file_to_run = dpg.get_value(user_data[0])
    dpg.configure_item("hyperparameter_prompt", show=False)

    config_path = "\"" + os.path.abspath(os.getcwd()) + "\\hyperparameter-configs\\" + hyperparameter_config_file_to_run + "\""
    runner.runTunerAndMlAgents(config_path, hyperParameterTuningData["mlagents"]["env_settings"]["env_path"], hyperParameterTuningData["realm_ai"]["behavior_name"])

def startGUI():
    dpg.create_context()
    dpg.create_viewport(title="REALM_AI", width=GLOBAL_WIDTH, height=GLOBAL_HEIGHT)
    dpg.set_global_font_scale(GLOBAL_FONT_SIZE)

    # Themes
    with dpg.theme(tag="__demo_hyperlinkTheme"):
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button, [0, 0, 0, 0])
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, [0, 0, 0, 0])
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, [29, 151, 236, 25])
            dpg.add_theme_color(dpg.mvThemeCol_Text, [29, 151, 236])

    # Able to browse previous runs and files
    with dpg.file_dialog(default_path="", directory_selector=False, show=False, id="file_dialog_id", height=150):
        dpg.add_file_extension(".*") # for any file - make it white 
        dpg.add_file_extension("", color=(255, 165, 0, 255)) # directories = orange
        dpg.add_file_extension(".yaml", color=(255, 0, 0, 255), custom_text="[Configuration]") # configuration files = red
        dpg.add_file_extension(".py", color=(0, 255, 0, 255), custom_text="[Python]") # python files = green

    # ML-Agents Prompt
    with dpg.window(label="ML-Agents Training", modal=True, pos=[GLOBAL_WIDTH/6, GLOBAL_HEIGHT/3] ,id="mlagents_prompt", show=False):
        dpg.add_text("This will start ml-agents training.\nChoose a ml-agents configuration file:")
        dpg.add_spacer(height=10)
        mlagents_config_file_to_run = dpg.add_combo(label="config_file", items=allMlAgentsConfigFiles)
        dpg.add_spacer(height=10)
        dpg.add_separator()
        dpg.add_spacer(height=1)

        with dpg.group(horizontal=True):
            dpg.add_button(label="Proceed", width=75, callback=run_training, user_data=[mlagents_config_file_to_run])
            dpg.add_button(label="Cancel", width=75, callback=lambda: dpg.configure_item("mlagents_prompt", show=False))

    # Tuner Prompt
    with dpg.window(label="Hyperparameter Tuning and Training", modal=True, pos=[GLOBAL_WIDTH/6, GLOBAL_HEIGHT/3] ,id="hyperparameter_prompt", show=False):
        dpg.add_text("This will start hyperparameter tuning and then start training.\nChoose a hyperparameter configuration file:")
        dpg.add_spacer(height=10)
        hyperparameter_config_file_to_run = dpg.add_combo(label="config_file", items=allHyperParameterTuningConfigFiles)
        dpg.add_spacer(height=10)
        dpg.add_separator()
        dpg.add_spacer(height=1)

        with dpg.group(horizontal=True):
            dpg.add_button(label="Proceed", width=75, callback=run_tune_and_training, user_data=[hyperparameter_config_file_to_run])
            dpg.add_button(label="Cancel", width=75, callback=lambda: dpg.configure_item("hyperparameter_prompt", show=False))

    # Main Window
    with dpg.window(label="Training Manager", width=GLOBAL_WIDTH-15, height=GLOBAL_HEIGHT-50, no_collapse=True, no_close=True):        

        # Introduction
        with dpg.group(horizontal=True):
            dpg.add_loading_indicator(circle_count=4)
            with dpg.group():
                dpg.add_text('Welcome to REALM_AI Training Manager!')
                with dpg.group(horizontal=True):
                    dpg.add_text("You can browse your existing training runs here: ")
                    dpg.add_button(label="Previous Runs", callback=lambda: dpg.show_item("file_dialog_id"))
        dpg.add_spacer(height=10)

        # ML-Agents Window
        with dpg.collapsing_header(label="ML-Agents Training", default_open=showMlAgents):
            dpg.add_text("Edit the values, press save, and then start training!", color=[232,163,33])
            with dpg.group(horizontal=True):
                dpg.add_text("For more information about the ml-agents training configuration files", color=[232,163,33])
                _hyperlink("click here", "https://github.com/Unity-Technologies/ml-agents/blob/main/docs/Training-Configuration-File.md")
            dpg.add_spacer(height=10)
            dpg.add_separator()

            dpg.add_text("Hyperparameters", color=[137,207,240])
            batch_size = dpg.add_input_int(label="batch_size", default_value=int(mlAgentsData["hyperparameters"]["batch_size"]), step=128, min_value=1, min_clamped=True)
            _help("Typical range (Continuous PPO): 512 - 5120")
            buffer_size = dpg.add_input_int(label="buffer_size", default_value=int(mlAgentsData["hyperparameters"]["buffer_size"]), step=128, min_value=32, min_clamped=True)
            _help("Typical range (PPO): 2048 - 409600")
            learning_rate = dpg.add_input_float(label="learning_rate", default_value=float(mlAgentsData["hyperparameters"]["learning_rate"]), format="%e", min_value=1e-9, min_clamped=True, max_value=0.99999, max_clamped=True)
            _help("Typical range: 1e-5 - 1e-3")
            beta = dpg.add_input_float(label="beta", default_value=float(mlAgentsData["hyperparameters"]["beta"]), format="%e", min_value=1e-9, min_clamped=True, max_value=0.49999, max_clamped=True)
            _help("Typical range: 1e-4 - 1e-2")
            epsilon = dpg.add_input_float(label="epsilon", default_value=float(mlAgentsData["hyperparameters"]["epsilon"]), min_value=1e-9, min_clamped=True, max_value=0.99999, max_clamped=True)
            _help("Typical range: 0.1 - 0.3")
            lambd = dpg.add_input_float(label="lambda", default_value=float(mlAgentsData["hyperparameters"]["lambd"]), min_value=1e-9, min_clamped=True, max_value=0.99999, max_clamped=True)
            _help("Typical range: 0.9 - 0.95")
            num_epoch = dpg.add_slider_int(label="num_epoch", default_value=float(mlAgentsData["hyperparameters"]["num_epoch"]), min_value=1, max_value=15)
            _help("Typical range: 3 - 10")
            lr_schedule = ["linear", "constant"]
            learning_rate_schedule = dpg.add_combo(label="learning_rate_schedule", items=lr_schedule, default_value=lr_schedule[0])
            _help("linear decays the learning_rate linearly, constant keeps the learning_rate constant")
            
            dpg.add_spacer(height=3)
            dpg.add_text("Network Settings", color=[137,207,240])
            normalize = dpg.add_checkbox(label="normalize", default_value=bool(mlAgentsData["network_settings"]["normalize"]))
            _help("Whether normalization is applied to the vector observation inputs")
            hidden_unit_values = [64, 128, 256, 512, 1024]
            hidden_units = dpg.add_combo(label="hidden_units", items=hidden_unit_values, default_value=hidden_unit_values[0])
            _help("Typical range: 32 - 512")
            num_layers = dpg.add_slider_int(label="num_layers", default_value=int(mlAgentsData["network_settings"]["num_layers"]), min_value=1, max_value=3)
            _help("Typical range: 1 - 3")

            dpg.add_spacer(height=3)
            dpg.add_text("Reward Signals")
            gamma = dpg.add_input_float(label="gamma", default_value=float(mlAgentsData["reward_signals"]["extrinsic"]["gamma"]), min_value=1e-9, min_clamped=True, max_value=0.99999, max_clamped=True)
            _help("Typical range: 0.8 - 0.995")
            strength = dpg.add_input_float(label="strength", default_value=float(mlAgentsData["reward_signals"]["extrinsic"]["strength"]), min_value=1e-9, min_clamped=True, max_value=0.99999, max_clamped=True)
            _help("Factor by which to multiply the reward given by the environment.")

            dpg.add_spacer(height=3)
            dpg.add_text("Other Settings", color=[137,207,240])
            keep_checkpoints = dpg.add_slider_int(label="keep_checkpoints", default_value=int(mlAgentsData["keep_checkpoints"]), min_value=1, max_value=10)
            _help("The maximum number of model checkpoints to keep")
            ml_max_steps = dpg.add_input_int(label="max_steps", default_value=float(mlAgentsData["max_steps"]), step=128, min_value=1, min_clamped=True)
            _help("Typical range: 5e5 - 1e7")
            time_horizon = dpg.add_input_int(label="time_horizon", default_value=int(mlAgentsData["time_horizon"]), step=128, min_value=1, min_clamped=True)
            _help("Typical range: 32 - 2048")
            summary_freq = dpg.add_input_int(label="summary_freq", default_value=int(mlAgentsData["summary_freq"]), step=128, min_value=1, min_clamped=True)
            _help("Number of experiences that needs to be collected before generating/displaying training stats")
            threaded = dpg.add_checkbox(label="threaded", default_value=bool(mlAgentsData["threaded"]))
            _help("Allow environments to step while updating the model. This might result in a training speedup, especially when using SAC. For best performance, set to false when using self-play")
            
            dpg.add_spacer(height=3)
            dpg.add_text("Configuration File Name", color=[137,207,240])
            mlagents_config_file_name = dpg.add_input_text(label="ml-agents_config_file_name", default_value=".yaml", width=400, hint="new config file name")

            dpg.add_spacer(height=20)

            with dpg.group(horizontal=True):
                dpg.add_button(label="Restore Defaults", callback=restore_ml_config, user_data=[batch_size, buffer_size, learning_rate, beta, epsilon, lambd, num_epoch, learning_rate_schedule, normalize, hidden_units, num_layers, gamma, strength, keep_checkpoints, ml_max_steps, time_horizon, summary_freq, threaded], small=True)
                dpg.add_spacer(width=8)
                dpg.add_button(label="Save ML-Agents Configuration", user_data=[batch_size, buffer_size, learning_rate, beta, epsilon, lambd, num_epoch, learning_rate_schedule, normalize, hidden_units, num_layers, gamma, strength, keep_checkpoints, ml_max_steps, time_horizon, summary_freq, threaded, mlagents_config_file_name], callback=edit_and_create_mlagents_config, small=True)
                dpg.add_spacer(width=8)
                dpg.add_button(label="Start Training", callback=prompt_show_ml_agents_config, user_data=[mlagents_config_file_to_run], small=True)
                dpg.add_spacer(height=30)

        # Hyperparameter Tuner Main Window
        with dpg.collapsing_header(label="Hyperparameter Tuning Configuration", default_open=showHyperParameter):
            with dpg.tab_bar(label="Tab bar"):
                # Basic Tab
                with dpg.tab(label="Basic"):
                    dpg.add_text("Basic Hyperparameter Tuning Configuration Values:\nEdit the values, press save, and then start training!", color=[232,163,33])
                    dpg.add_spacer(height=10)

                    hyperparameter_config_file_name = dpg.add_input_text(label="hyperparameter_config_file_name", default_value=".yaml", width=400, hint="new config file name")
                    env_path = dpg.add_input_text(label="env_path", default_value=str(hyperParameterTuningData["mlagents"]["env_settings"]["env_path"]), width=400, hint="env_path of ml-agents")
                    behavior_name = dpg.add_input_text(label="behavior_name", default_value=str(hyperParameterTuningData["realm_ai"]["behavior_name"]), width=400, hint="unity game project name")
                    max_steps = dpg.add_input_int(label="max_steps", default_value=int(hyperParameterTuningData["mlagents"]["default_settings"]["max_steps"]), step=1000, min_value=1, max_value=1e9)
                    dpg.add_spacer(height=25)

                    with dpg.group(horizontal=True):
                        dpg.add_button(label="Restore Defaults", callback=restore_hyperparameter_config, user_data=[env_path, behavior_name, max_steps], small=True)
                        dpg.add_spacer(width=8)
                        dpg.add_button(label="Save Hyperparameter Configuration", callback=edit_and_create_hyperparameter_config, user_data=[env_path, behavior_name, max_steps, hyperparameter_config_file_name], small=True)
                        dpg.add_spacer(width=8)
                        dpg.add_button(label="Start Hyperparameter Tuning and Training", callback=prompt_show_hyperparameter_config, user_data=[hyperparameter_config_file_to_run], small=True)

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
    # Command line Arguments
    parser = argparse.ArgumentParser(description='Python GUI to launch ML-Agents and Hyperparameter Tuning')
    parser.add_argument('--env-path', type=str, default=None, help="Path to environment. If specified, overrides env_path in the config file")
    parser.add_argument('--behavior-name', type=str, default=None, help='Name of behaviour. This can be found under the agent\'s "Behavior Parameters" component in the inspector of Unity')
    parser.add_argument('--mlagents', action='store_true')
    parser.add_argument('--hyperparameter', action='store_true')

    # ML Agents Data
    mlAgentsConfigFile = "ml-agents-configs/" + MASTER_MLAGENTS_FILE
    mlAgentsData = loadMlAgentsConfig(mlAgentsConfigFile)["default_settings"]

    # Hyper Parameter Data
    hyperParameterConfigFile = "hyperparameter-configs/" + MASTER_HYPERPARAMETERS_FILE
    hyperParameterTuningData = loadHyperParameterConfig(hyperParameterConfigFile)

    # Override loaded in data with the command line arguments
    args = parser.parse_args()
    if args.behavior_name != None:
        hyperParameterTuningData["realm_ai"]["behavior_name"] = args.behavior_name
    if args.env_path != None:
        hyperParameterTuningData["mlagents"]["env_settings"]["env_path"] = args.env_path
    if args.mlagents:
        showMlAgents = True
    if args.hyperparameter:
        showHyperParameter = True

    defaultMlAgentsData = copy.deepcopy(mlAgentsData)
    defaultHyperParameterTuningData = copy.deepcopy(hyperParameterTuningData)

    startGUI()