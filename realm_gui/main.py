import argparse
import copy
import os
import shutil
import webbrowser
import subprocess
from datetime import datetime
from importlib.resources import path
import multiprocessing

import dearpygui.dearpygui as dpg
import yaml

import realm_gui

# ML-Agents Variables
mlAgentsData = {}
defaultMlAgentsData = {}
allMlAgentsConfigFiles = []
showMlAgents = False

# Hyper Parameter Tuner Variables
hyperParameterTuningData = {}
defaultHyperParameterTuningData = {}
allHyperParameterTuningConfigFiles = []
showHyperParameter = False

# Dashboard Variables
showDashboard = False
dashboardBackendProcess = None

# results directory
resultsDir = "results"
runId = "ppo"

# some dropdown options
training_resume_options = ["No", "Initialize from previous run", "Continue previous run"]
training_resume_help = ("Choose \"No\" to start a completed new training run. \n"
    "Choose \"Initialize from previous run\" to start a new training run, \nbut to start with the trained model from a previous run. \n"
    "Choose \"Continue previous run\" to continue training a previous training run. \n"
    "Note that if \"Continue previous run\" is chosen, a new results directory will not be created -- \nthe results will continue to be outputted into the previuos run's directory.")

tuning_resume_options = ["No", "Continue previous run"]
tuning_resume_help = ("Choose \"No\" to start a completed new training run. \n"
    "Choose \"Continue previous run\" to continue training a previous training run. \n"
    "Note that if \"Continue previous run\" is chosen, a new results directory will not be created -- \nthe results will continue to be outputted into the previuos run's directory.")


# Global Window Variables
GLOBAL_WIDTH = 1000
GLOBAL_HEIGHT = 800
GLOBAL_FONT_SIZE = 1.15

def runTunerAndMlAgents(configPath: str, envPath: str, previous_run: str, start_new_run):
    global resultsDir, runId

    if start_new_run:
        resultsDirParameter = "--output-path=\"%s\"" % os.path.join(resultsDir, runId)
    else:
        resultsDirParameter = "--output-path=\"%s\"" % previous_run
    configPathParameter = "--config-path=\"%s\"" % configPath
    envPathParameter = "--env-path=\"%s\"" % envPath
    print("realm-tune %s %s %s" % (configPathParameter, envPathParameter, resultsDirParameter))
    os.system("realm-tune %s %s %s" % (configPathParameter, envPathParameter, resultsDirParameter))

"""
mlagents-learn <trainer-config-file> --env=<env_name> --run-id=<run-identifier>

<trainer-config-file> is the file path of the trainer configuration YAML.
<env_name>(Optional) is the name (including path) of your Unity executable containing the agents to be trained. If <env_name> is not passed, the training will happen in the Editor.
<run-identifier> is a unique name you can use to identify the results of your training runs.

https://github.com/Unity-Technologies/ml-agents/blob/main/docs/Training-ML-Agents.md
"""
def runMLagents(configPath: str, previous_run: str, start_new_run):
    global resultsDir, runId

    if start_new_run:
        # start new run
        # if previous run is specified, intiailized from it
        runIdParameter = "--run-id=" + runId
        resultsDirParameter = "--results-dir=\"%s\"" % resultsDir
        initializeFromParameter = "" if not previous_run else ("--initialize-from=\"%s\"" % previous_run)
        
        print("mlagents-learn \"%s\" %s %s %s --force" % (configPath, runIdParameter, resultsDirParameter, initializeFromParameter))
        os.system("mlagents-learn \"%s\" %s %s %s --force" % (configPath, runIdParameter, resultsDirParameter, initializeFromParameter))
    else:
        # try to continue the previous run
        runId = os.path.basename(os.path.normpath(previous_run))
        resultsDir = os.path.dirname(os.path.normpath(previous_run))
        runIdParameter = "--run-id=" + runId
        resultsDirParameter = "--results-dir=\"%s\"" % resultsDir
        print("mlagents-learn \"%s\" %s %s --resume" % (configPath,  runIdParameter, resultsDirParameter))
        os.system("mlagents-learn \"%s\" %s %s --resume" % (configPath, runIdParameter, resultsDirParameter))

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

def _verifyEnvPath(envPath):
    if envPath is None or envPath == "":
        return False

    return True

def restore_ml_config(sender, app_data, user_data):
    global mlAgentsData, defaultMlAgentsData

    print("ml configuration restored") # debugging log

    mlAgentsData["hyperparameters"]["batch_size"] = int(defaultMlAgentsData["hyperparameters"]["batch_size"])
    mlAgentsData["hyperparameters"]["buffer_size"] = int(defaultMlAgentsData["hyperparameters"]["buffer_size"])
    mlAgentsData["hyperparameters"]["learning_rate"] = float(defaultMlAgentsData["hyperparameters"]["learning_rate"])
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

    dpg.set_value(user_data[3], mlAgentsData["hyperparameters"]["learning_rate_schedule"])

    dpg.set_value(user_data[4], mlAgentsData["network_settings"]["normalize"])
    dpg.set_value(user_data[5], mlAgentsData["network_settings"]["hidden_units"])
    dpg.set_value(user_data[6], mlAgentsData["network_settings"]["num_layers"])

    dpg.set_value(user_data[7], mlAgentsData["reward_signals"]["extrinsic"]["gamma"])
    dpg.set_value(user_data[8], mlAgentsData["reward_signals"]["extrinsic"]["strength"])

    dpg.set_value(user_data[9], mlAgentsData["keep_checkpoints"])
    dpg.set_value(user_data[10], mlAgentsData["max_steps"])
    dpg.set_value(user_data[11], mlAgentsData["time_horizon"])
    dpg.set_value(user_data[12], mlAgentsData["summary_freq"])
    dpg.set_value(user_data[13], mlAgentsData["threaded"])

def restore_hyperparameter_config(sender, app_data, user_data):
    global hyperParameterTuningData, defaultHyperParameterTuningData

    print("hyperparameter configuration restored") # debugging log

    hyperParameterTuningData["realm_ai"]["env_path"] = ""
    hyperParameterTuningData["mlagents"]["default_settings"]["max_steps"] = defaultHyperParameterTuningData["mlagents"]["default_settings"]["max_steps"]
    hyperParameterTuningData["realm_ai"]["total_trials"] = defaultHyperParameterTuningData["realm_ai"]["total_trials"]
    hyperParameterTuningData["realm_ai"]["algorithm"] = defaultHyperParameterTuningData["realm_ai"]["algorithm"]

    dpg.set_value(user_data[0], hyperParameterTuningData["realm_ai"]["env_path"])
    dpg.set_value(user_data[1], hyperParameterTuningData["mlagents"]["default_settings"]["max_steps"])
    dpg.set_value(user_data[2], hyperParameterTuningData["realm_ai"]["total_trials"])
    dpg.set_value(user_data[3], hyperParameterTuningData["realm_ai"]["algorithm"])
    dpg.set_value(user_data[4], multiprocessing.cpu_count())
    dpg.set_value(user_data[5], hyperParameterTuningData["mlagents"]["default_settings"]["max_steps"])
    dpg.set_value(user_data[6], hyperParameterTuningData["mlagents"]["default_settings"]["trainer_type"])


def algorithm_chooser_cb(sender, app_data, user_data):

    def sac_cb():
        dpg.set_value("training_algorithm_dropdown", 'sac')
        dpg.configure_item("algorithm_chooser", show=False)

    def ppo_cb():
        dpg.set_value("training_algorithm_dropdown", 'ppo')
        dpg.configure_item("algorithm_chooser", show=False)

    dpg.configure_item("algorithm_chooser", show=True)
    
    try:
        import torch
        if torch.cuda.is_available():
            device = torch.device("cuda")
            _ = torch.empty((2,2), device=device)
            recommend_sac=True
        else: recommend_sac = False
    except:
        recommend_sac = False
    
    dpg.configure_item("algorithm_chooser_loading_bar", show=False)
    dpg.configure_item("algorithm_chooser_recommendation", show=True)
    if recommend_sac:
        dpg.configure_item("algorithm_chooser_recommend_sac", show=True)
        dpg.configure_item("algorithm_chooser_button", label="Use SAC", callback=sac_cb)
    else:
        dpg.configure_item("algorithm_chooser_recommend_ppo", show=True)
        dpg.configure_item("algorithm_chooser_button", label="Use PPO", callback=ppo_cb)


def edit_and_create_mlagents_config(sender, app_data, user_data):
    global mlAgentsData

    print("ml-agents configuration saved") # debugging log

    # Overwrite existing ml-agents config file
    mlAgentsData["hyperparameters"]["batch_size"] = dpg.get_value(user_data[0])
    mlAgentsData["hyperparameters"]["buffer_size"] = dpg.get_value(user_data[1])
    mlAgentsData["hyperparameters"]["learning_rate"] = dpg.get_value(user_data[2])

    mlAgentsData["hyperparameters"]["learning_rate_schedule"] = dpg.get_value(user_data[3])

    mlAgentsData["network_settings"]["normalize"] = dpg.get_value(user_data[4])
    mlAgentsData["network_settings"]["hidden_units"] = int(dpg.get_value(user_data[5]))
    mlAgentsData["network_settings"]["num_layers"] = dpg.get_value(user_data[6])

    mlAgentsData["reward_signals"]["extrinsic"]["gamma"] = dpg.get_value(user_data[7])
    mlAgentsData["reward_signals"]["extrinsic"]["strength"] = dpg.get_value(user_data[8])

    mlAgentsData["keep_checkpoints"] = dpg.get_value(user_data[9])
    mlAgentsData["max_steps"] = dpg.get_value(user_data[10])
    mlAgentsData["time_horizon"] = dpg.get_value(user_data[11])
    mlAgentsData["summary_freq"] = dpg.get_value(user_data[12])
    mlAgentsData["threaded"] = dpg.get_value(user_data[13])

    if user_data[15] is not None:
        if 'env_settings' not in mlAgentsData:
            mlAgentsData['env_settings'] = {}
        mlAgentsData["env_settings"]["env_args"] = ["-ffmpeg-path", user_data[15]]

    reformattedMlAgentsData = {}
    reformattedMlAgentsData["default_settings"] = mlAgentsData

    if len(dpg.get_value(user_data[14])) != 0 and dpg.get_value(user_data[14]) != ".yaml":
        newConfigFile = dpg.get_value(user_data[14])
    else:
        newConfigFile = "config.yaml"

    with path(realm_gui, 'ml_agents_configs') as f:
        mlAgentsConfigFile = os.path.join(f, newConfigFile)
        with open(mlAgentsConfigFile, 'w') as outfile:
            yaml.dump(reformattedMlAgentsData, outfile, sort_keys=False)

    print("new ml-agents configuration created") # debugging log

def edit_and_create_hyperparameter_config(sender, app_data, user_data):
    global hyperParameterTuningData

    if dpg.get_value(user_data[8])=='sac':
        with path(realm_gui, 'hyperparameter_configs') as f:
            hyperParameterConfigFile = os.path.join(f, 'bayes_sac.yaml')
            hyperParameterTuningData = loadHyperParameterConfig(hyperParameterConfigFile)
    # Overwrite existing hyperparameter config file
    hyperParameterTuningData["realm_ai"]["env_path"] = dpg.get_value(user_data[0])
    hyperParameterTuningData["realm_ai"]["full_run_after_tuning"]["max_steps"] = int(dpg.get_value(user_data[1]))
    hyperParameterTuningData["realm_ai"]["total_trials"] = int(dpg.get_value(user_data[2]))
    hyperParameterTuningData["realm_ai"]["warmup_trials"] = int(dpg.get_value(user_data[2])) // 3
    hyperParameterTuningData["realm_ai"]["algorithm"] = dpg.get_value(user_data[3])
    
    if 'env_settings' not in hyperParameterTuningData['mlagents']:
        hyperParameterTuningData['mlagents']['env_settings'] = {}
    
    if user_data[5] is not None:
        hyperParameterTuningData['mlagents']["env_settings"]["env_args"] = ["-ffmpeg-path", user_data[5]]

    hyperParameterTuningData['mlagents']['env_settings']['num_envs'] = int(dpg.get_value(user_data[6]))
    hyperParameterTuningData["mlagents"]["default_settings"]["max_steps"] = int(dpg.get_value(user_data[7]))
    if not _verifyEnvPath(hyperParameterTuningData["realm_ai"]["env_path"]):
        dpg.configure_item("env_path_validation_prompt", show=True)
        return 

    print("hyperparameter configuration saved") # debugging log

    if len(dpg.get_value(user_data[4])) != 0 and dpg.get_value(user_data[4]) != ".yaml":
        newConfigFile = dpg.get_value(user_data[4])
    else:
        newConfigFile = "config.yaml"

    with path(realm_gui, 'hyperparameter_configs') as f:
        hyperParameterConfigFile = os.path.join(f, newConfigFile)
        with open(hyperParameterConfigFile, 'w') as outfile:
            yaml.dump(hyperParameterTuningData, outfile, sort_keys=False)

    print("new hyperparameter configuration created") # debugging log

def prompt_show_hyperparameter_config(sender, app_data, user_data):
    global allHyperParameterTuningConfigFiles

    if not _verifyEnvPath(dpg.get_value(user_data[0])):
        dpg.configure_item("env_path_validation_prompt", show=True)
        return 

    # Get all possible config files in current directory to show up
    allHyperParameterTuningConfigFiles = [] # reset list
    with path(realm_gui, 'hyperparameter_configs') as directory:
        try:
            for file in os.listdir(directory):
                if file.endswith(".yaml"):
                    allHyperParameterTuningConfigFiles.append(file)

            allHyperParameterTuningConfigFiles.sort()
        except OSError as e:
            print("Error: %s : %s" % (directory, e.strerror))
            allHyperParameterTuningConfigFiles = []

    # Prompt To Appear - user_data[1] = prompt dropdown component which dynamically changes it list contents
    dpg.configure_item(user_data[1], items=allHyperParameterTuningConfigFiles)
    dpg.configure_item("hyperparameter_prompt", show=True)

def prompt_show_ml_agents_config(sender, app_data, user_data):
    global allMlAgentsConfigFiles

    # Get all possible config files in current directory to show up
    allMlAgentsConfigFiles = [] # reset list

    with path(realm_gui, 'ml_agents_configs') as directory:
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
    previous_run = dpg.get_value(user_data[1])
    resume_option = dpg.get_value(user_data[2])
    dpg.configure_item("mlagents_prompt", show=False)

    start_new_run = True
    if resume_option == training_resume_options[0]:
        previous_run = ""
        start_new_run = True
    elif resume_option == training_resume_options[1]:
        start_new_run = True
    else:
        start_new_run = False

    with path(realm_gui, 'ml_agents_configs') as f:
        mlAgentsConfigFile = os.path.join(f, mlagents_config_file_to_run)
        runMLagents(mlAgentsConfigFile, previous_run, start_new_run)


# user_data[0] = hyperparameter_config_file_to_run
def run_tune_and_training(sender, app_data, user_data):
    global hyperParameterTuningData

    hyperparameter_config_file_to_run = dpg.get_value(user_data[0])
    previous_run = dpg.get_value(user_data[1])
    resume_option = dpg.get_value(user_data[2])
    dpg.configure_item("hyperparameter_prompt", show=False)

    start_new_run = True
    if resume_option == training_resume_options[0]:
        previous_run = ""
        start_new_run = True
    else:
        start_new_run = False
    
    with path(realm_gui, 'hyperparameter_configs') as f:
        hyperParameterConfigFile = os.path.join(f, hyperparameter_config_file_to_run)
        runTunerAndMlAgents(hyperParameterConfigFile, hyperParameterTuningData["realm_ai"]["env_path"], previous_run, start_new_run)

def mlagents_resume_dropdown_update(sender, app_data, user_data):
    choice = dpg.get_value(sender)
    dpg.configure_item("training_previous_run_group", show=(choice != training_resume_options[0]))

def hyperparameter_resume_dropdown_update(sender, app_data, user_data):
    choice = dpg.get_value(sender)
    dpg.configure_item("tuning_previous_run_group", show=(choice != tuning_resume_options[0]))


def start_dashboard_backend(sender, app_data, user_data):
    global dashboardBackendProcess
    dashboardBackendProcess = subprocess.Popen("realm-report", creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    dpg.set_value("dashboardBackendStatus", "Dashboard backend process is running.")
    
def stop_dashboard_backend(sender, app_data, user_data):
    global dashboardBackendProcess
    if dashboardBackendProcess != None:
        dashboardBackendProcess.terminate()
        dashboardBackendProcess = None
        dpg.set_value("dashboardBackendStatus", "Dashboard backend process is stopped.")
    
def open_dashboard(sender, app_data, user_data):
    start_dashboard_backend(sender, app_data, user_data)
    webbrowser.open("http://localhost:5000")

def on_start_gui(sender, app_data, user_data):
    global showDashboard
    if showDashboard:
        start_dashboard_backend(sender, app_data, user_data)
        open_dashboard(sender, app_data, user_data)

def on_exit_gui(sender, app_data, user_data):
    stop_dashboard_backend(sender, app_data, user_data)

def startGUI(args : argparse.Namespace):
    global showMlAgents, showHyperParameter, showDashboard, mlAgentsData, hyperParameterTuningData, allMlAgentsConfigFiles, allHyperParameterTuningConfigFiles

    dpg.create_context()
    dpg.create_viewport(title="REALM_AI Training Manager", width=GLOBAL_WIDTH, height=GLOBAL_HEIGHT)
    dpg.set_global_font_scale(GLOBAL_FONT_SIZE)

    # Themes
    with dpg.theme(tag="__demo_hyperlinkTheme"):
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button, [0, 0, 0, 0])
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, [0, 0, 0, 0])
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, [29, 151, 236, 25])
            dpg.add_theme_color(dpg.mvThemeCol_Text, [29, 151, 236])

    # Values
    with dpg.value_registry():
        dpg.add_string_value(default_value="Dashboard backend process has not been started.", tag="dashboardBackendStatus")

    # Able to browse previous runs and files
    with dpg.file_dialog(default_path="", directory_selector=False, show=False, id="file_dialog_id", height=150):
        dpg.add_file_extension(".*") # for any file - make it white 
        dpg.add_file_extension("", color=(255, 165, 0, 255)) # directories = orange
        dpg.add_file_extension(".yaml", color=(255, 0, 0, 255), custom_text="[Configuration]") # configuration files = red
        dpg.add_file_extension(".py", color=(0, 255, 0, 255), custom_text="[Python]") # python files = green

    # ML-Agents Prompt
    with dpg.window(label="ML-Agents Training", modal=True, pos=[GLOBAL_WIDTH/6, GLOBAL_HEIGHT/3], height=300, id="mlagents_prompt", show=False):
        dpg.add_text("This will start ml-agents training.\nChoose a ml-agents configuration file:")
        dpg.add_spacer(height=10)
        mlagents_config_file_to_run = dpg.add_combo(label="config_file", items=allMlAgentsConfigFiles)
        dpg.add_spacer(height=10)
        
        dpg.add_text("Continue previous run?")
        with dpg.group(horizontal=True):
            resume_dropdown = dpg.add_combo(label="", items=training_resume_options, default_value=training_resume_options[0], width=200, tag="training_resume_dropdown", callback=mlagents_resume_dropdown_update)
            _help(training_resume_help)
        dpg.add_spacer(height=10)
                
        with dpg.group(id="training_previous_run_group", show=False):
            dpg.add_text("Previous Run Results Directory")
            with dpg.group(horizontal=True):
                previous_run = dpg.add_input_text(label="", id="previous_run_input")
                _help("Enter the path of the directory containing the results of the previous tuning and run.")
            dpg.add_spacer(height=10)

        dpg.add_separator()
        dpg.add_spacer(height=1)

        with dpg.group(horizontal=True):
            dpg.add_button(label="Proceed", width=75, callback=run_training, user_data=[mlagents_config_file_to_run, previous_run, resume_dropdown])
            dpg.add_button(label="Cancel", width=75, callback=lambda: dpg.configure_item("mlagents_prompt", show=False))

    # Tuner Prompt
    with dpg.window(label="Hyperparameter Tuning and Training", modal=True, pos=[GLOBAL_WIDTH/6, GLOBAL_HEIGHT/3], height=300, id="hyperparameter_prompt", show=False):
        dpg.add_text("This will start hyperparameter tuning and then start training.\nChoose a hyperparameter configuration file:")
        dpg.add_spacer(height=10)
        hyperparameter_config_file_to_run = dpg.add_combo(label="config_file", items=allHyperParameterTuningConfigFiles)
        dpg.add_spacer(height=10)
        
        dpg.add_text("Continue previous run?")
        with dpg.group(horizontal=True):
            resume_dropdown = dpg.add_combo(label="", items=tuning_resume_options, default_value=tuning_resume_options[0], width=200, tag="tuning_resume_dropdown", callback=hyperparameter_resume_dropdown_update)
            _help(tuning_resume_help)
        dpg.add_spacer(height=10)
                
        with dpg.group(id="tuning_previous_run_group", show=False):
            dpg.add_text("Previous Run Results Directory")
            with dpg.group(horizontal=True):
                previous_run = dpg.add_input_text(label="", id="tuning_previous_run_input")
                _help("Enter the path of the directory containing the results of the previous tuning and training run.")
            dpg.add_spacer(height=10)

        dpg.add_separator()
        dpg.add_spacer(height=1)

        with dpg.group(horizontal=True):
            dpg.add_button(label="Proceed", width=75, callback=run_tune_and_training, user_data=[hyperparameter_config_file_to_run, previous_run, resume_dropdown])
            dpg.add_button(label="Cancel", width=75, callback=lambda: dpg.configure_item("hyperparameter_prompt", show=False))

    # PPO/SAC algorithm chooser prompt
    with dpg.window(label="Training Algorithm Recommendation", modal=True, pos=[20, GLOBAL_HEIGHT//3], width=GLOBAL_WIDTH-40, height=GLOBAL_HEIGHT//3, id="algorithm_chooser", show=False):
        # dpg.configure_item("loading_screen", show=True)
        dpg.add_loading_indicator(tag="algorithm_chooser_loading_bar", circle_count=8)
        with dpg.group(tag="algorithm_chooser_recommendation", show=False):
            dpg.add_text("A cuda-enabled card has been detected. We recommend the Soft Actor Critic(SAC) algorithm,\nespecially if the environment is heavier/slower.", tag="algorithm_chooser_recommend_sac", show=False)
            dpg.add_text("A cuda-enabled card has not been detected. We recommend the Proximal Policy Optimization(PPO) algorithm.", tag="algorithm_chooser_recommend_ppo", show=False)
            _hyperlink("Click here for more information", "https://github.com/Unity-Technologies/ml-agents/blob/main/docs/ML-Agents-Overview.md#deep-reinforcement-learning")
            dpg.add_spacer(height=10)
            with dpg.group(horizontal=True):
                dpg.add_button(label="", width=120, tag="algorithm_chooser_button")
                dpg.add_button(label="Cancel", width=80, callback=lambda: dpg.configure_item("algorithm_chooser", show=False))
            #     dpg.add_spacer(width=20)
            #     dpg.add_button(label="No, continue with PPO", width=175, callback=close_window_and_call_cb)

    # Main Window
    with dpg.window(label="Training Manager", tag="Main Window", width=GLOBAL_WIDTH-15, height=GLOBAL_HEIGHT-50, no_collapse=True, no_close=True):        

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
            dpg.add_text("Reward Signals", color=[137,207,240])
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
                dpg.add_button(label="Restore Defaults", callback=restore_ml_config, user_data=[batch_size, buffer_size, learning_rate, learning_rate_schedule, normalize, hidden_units, num_layers, gamma, strength, keep_checkpoints, ml_max_steps, time_horizon, summary_freq, threaded], small=True)
                dpg.add_spacer(width=8)
                dpg.add_button(label="Save ML-Agents Configuration", user_data=[batch_size, buffer_size, learning_rate, learning_rate_schedule, normalize, hidden_units, num_layers, gamma, strength, keep_checkpoints, ml_max_steps, time_horizon, summary_freq, threaded, mlagents_config_file_name, args.ffmpeg_path], callback=edit_and_create_mlagents_config, small=True)
                dpg.add_spacer(width=8)
                dpg.add_button(label="Start Training", callback=prompt_show_ml_agents_config, user_data=[mlagents_config_file_to_run], small=True)
                dpg.add_spacer(height=30)

        # Hyperparameter Tuner Main Window
        with dpg.collapsing_header(label="Hyperparameter Tuning Configuration", default_open=showHyperParameter):
            dpg.add_text("Basic Hyperparameter Tuning Configuration Values:\nEdit the values, press save, and then start training!", color=[232,163,33])
            dpg.add_spacer(height=10)
            
            tuning_algorithms = ["bayes", "random"]
            tuning_algorithm = dpg.add_combo(label="tuning algorithm", items=tuning_algorithms, default_value=tuning_algorithms[0])
            _help("Hyperparameter tuning algorithm")
            env_path = dpg.add_input_text(label="env_path", default_value=hyperParameterTuningData["realm_ai"]["env_path"], width=400, hint="env_path of ml-agents")
            _help("Env path of Ml-Agents")
            num_envs = dpg.add_input_int(label="num_envs", default_value=multiprocessing.cpu_count(), step=1, min_value=1, min_clamped=True)
            _help("Number of parallel environments. Set to number of logical cores of system by default.")
            total_trials = dpg.add_input_int(label="total_trials", default_value=int(hyperParameterTuningData["realm_ai"]["total_trials"]), step=1, min_value=1, min_clamped=True)
            _help("Number of hyperparameter tuning trials")
            tuning_steps = dpg.add_input_int(label="tuning_steps", default_value=int(hyperParameterTuningData["mlagents"]["default_settings"]["max_steps"]), step=10000,min_value=1, min_clamped=True)
            _help("Number of training steps per hyperparameter tuning run")

            dpg.add_spacer(height=3)
            dpg.add_text("Full Training Run Settings", color=[137,207,240])
            dpg.add_text("A full training run is automatically initiated after hyperparameter tuning.\nThe full training run automatically uses the best hyperparameters found from the tuning process.", color=[232,163,33], )
            full_run_max_steps = dpg.add_input_int(label="training_steps", default_value=int(hyperParameterTuningData["realm_ai"]["full_run_after_tuning"]["max_steps"]), step=10000, min_value=1, min_clamped=True)
            _help("Number of training steps for the full run")

            dpg.add_spacer(height=3)
            dpg.add_text("Configuration File Name", color=[137,207,240])
            hyperparameter_config_file_name = dpg.add_input_text(label="hyperparameter_config_file_name", default_value=".yaml", width=400, hint="new config file name")

            dpg.add_spacer(height=3)
            dpg.add_text("ML-Agent Settings", color=[137,207,240])
            training_algorithms = ["ppo", "sac"]
            with dpg.group(horizontal=True):
                training_algorithm = dpg.add_combo(label="RL algorithm", items=training_algorithms, default_value=training_algorithms[0], width=200, tag="training_algorithm_dropdown")
                _help("Reinforcement Learning algorithm that will be used. The options are Proximal Policy Optimization(PPO) and Soft Actor Critic(SAC)")
                dpg.add_button(label="Help Me Decide", callback=algorithm_chooser_cb)
            dpg.add_spacer(height=25)

            with dpg.group(horizontal=True):
                dpg.add_button(label="Restore Defaults", callback=restore_hyperparameter_config, user_data=[env_path, full_run_max_steps, total_trials, tuning_algorithm,num_envs, tuning_steps, training_algorithm], small=True)
                dpg.add_spacer(width=8)
                dpg.add_button(label="Save Hyperparameter Configuration", callback=edit_and_create_hyperparameter_config, user_data=[env_path, full_run_max_steps, total_trials, tuning_algorithm, hyperparameter_config_file_name, args.ffmpeg_path, num_envs, tuning_steps, training_algorithm], small=True)
                dpg.add_spacer(width=8)
                dpg.add_button(label="Start Hyperparameter Tuning and Training", callback=prompt_show_hyperparameter_config, user_data=[env_path, hyperparameter_config_file_to_run], small=True)
            dpg.add_spacer(height=30)

        #  Main Window
        with dpg.collapsing_header(label="Dashboard", default_open=showDashboard):
            dpg.add_text("Dashboard:\nStart the dashboard backend to view your results in any web browser!", color=[232,163,33])
            dpg.add_spacer(height=10)

            dpg.add_text(source="dashboardBackendStatus", color=[232,163,33])
            dpg.add_spacer(height=25)

            with dpg.group(horizontal=True):
                dpg.add_button(label="Open Dashboard", callback=open_dashboard, small=True)                
                dpg.add_spacer(height=30)

        # Env Path Validation Error Prompt
        with dpg.window(label="ERROR", modal=True, pos=[GLOBAL_WIDTH/6, GLOBAL_HEIGHT/3] ,id="env_path_validation_prompt", show=False):
            dpg.add_text("ERROR: env_path can not be empty. It must have a value.")
            dpg.add_spacer(height=10)
            dpg.add_button(label="Continue", width=75, callback=lambda: dpg.configure_item("env_path_validation_prompt", show=False))

    dpg.set_frame_callback(1, on_start_gui)
    dpg.set_exit_callback(on_exit_gui)

    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("Main Window", True)
    dpg.start_dearpygui()
    dpg.destroy_context()

def main():
    global showMlAgents, showHyperParameter, showDashboard, mlAgentsData, hyperParameterTuningData, defaultMlAgentsData, defaultHyperParameterTuningData, resultsDir, runId

    # Master Configuration Files
    MASTER_MLAGENTS_FILE = "ppo.yaml"
    MASTER_HYPERPARAMETERS_FILE = "bayes.yaml"
    
    # Command line Arguments
    parser = argparse.ArgumentParser(description='Python GUI to launch ML-Agents and Hyperparameter Tuning')
    parser.add_argument('--env-path', type=str, default=None, help="Path to environment. If specified, overrides env_path in the config file")
    parser.add_argument('--behavior-name', type=str, default=None, help='Name of behaviour. This can be found under the agent\'s "Behavior Parameters" component in the inspector of Unity')
    parser.add_argument('--mlagents', action='store_true')
    parser.add_argument('--hyperparameter', action='store_true')
    parser.add_argument('--dashboard', action='store_true')
    parser.add_argument('--results-dir', type=str, default="results", help="Results directory for training results to be outputed to.")
    parser.add_argument('--run-id', type=str, default="ppo", help="Identifier for the training run. Used to name the subdirectory for the training data within the results directory")
    parser.add_argument('--ffmpeg-path', type=str, default=None)

    # ML Agents Data
    with path(realm_gui, 'ml_agents_configs') as f:
        mlAgentsConfigFile = os.path.join(f, MASTER_MLAGENTS_FILE)
        mlAgentsData = loadMlAgentsConfig(mlAgentsConfigFile)["default_settings"]

    # Hyper Parameter Data
    with path(realm_gui, 'hyperparameter_configs') as f:
        hyperParameterConfigFile = os.path.join(f, MASTER_HYPERPARAMETERS_FILE)
        hyperParameterTuningData = loadHyperParameterConfig(hyperParameterConfigFile)

    # Override loaded in data with the command line arguments
    args = parser.parse_args()
    if args.behavior_name != None:
        hyperParameterTuningData["realm_ai"]["behavior_name"] = args.behavior_name
    if args.env_path != None:
        hyperParameterTuningData["realm_ai"]["env_path"] = args.env_path
    else:
        hyperParameterTuningData["realm_ai"]["env_path"] = ""
    if args.mlagents:
        showMlAgents = True
    if args.hyperparameter:
        showHyperParameter = True
    if args.dashboard:
        showDashboard = True
    resultsDir = args.results_dir
    runId = args.run_id

    defaultMlAgentsData = copy.deepcopy(mlAgentsData)
    defaultHyperParameterTuningData = copy.deepcopy(hyperParameterTuningData)

    startGUI(args)

if __name__ == "__main__":
    main()