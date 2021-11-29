from datetime import datetime
import os
import shutil
import subprocess

def runTunerAndMlAgents(configPath: str, envPath: str, behaviorName: str):
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    configPathParameter = "--config-path=" + configPath
    envPathParameter = "--env-path=\"" + envPath + "\""
    behaviorNameParameter = "--behavior-name=" + behaviorName
    print("realm-tune %s %s %s" % (configPathParameter, envPathParameter, behaviorNameParameter))
    os.system("realm-tune %s %s %s" % (configPathParameter, envPathParameter, behaviorNameParameter))
    # shutil.move("results/" + behaviorName + "-" + now, "results/Train")

"""
mlagents-learn <trainer-config-file> --env=<env_name> --run-id=<run-identifier>

<trainer-config-file> is the file path of the trainer configuration YAML.
<env_name>(Optional) is the name (including path) of your Unity executable containing the agents to be trained. If <env_name> is not passed, the training will happen in the Editor.
<run-identifier> is a unique name you can use to identify the results of your training runs.

https://github.com/Unity-Technologies/ml-agents/blob/main/docs/Training-ML-Agents.md
"""
def runMLagents(configPath: str, behaviorName: str):
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    behaviorNameParameter = "--run-id=" + behaviorName + "-" + now
    print("mlagents-learn %s %s --force" % (configPath, behaviorNameParameter))
    os.system("mlagents-learn %s %s --force" % (configPath, behaviorNameParameter))
    shutil.move("results/" + behaviorName + "-" + now, "results/Train")