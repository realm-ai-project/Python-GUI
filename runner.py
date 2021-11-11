from datetime import datetime
import os
import subprocess

def runTunerAndMlAgents(configPath: str, envPath: str, behaviorName: str):
    folder = "results/Tune_Train/" + behaviorName + "-" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(folder)

    configPathParameter = "--config-path=" + configPath
    envPathParameter = "--env-path=" + envPath
    behaviorNameParameter = "--behavior-name=" + behaviorName
    print("realm-tune %s %s %s" % (configPathParameter, envPathParameter, behaviorNameParameter))

    # Ken uncomment and test, check if the command I output to std out above is correct
    # subprocess.run(["realm-tune", configPathParameter, envPathParameter, behaviorNameParameter])

"""
mlagents-learn <trainer-config-file> --env=<env_name> --run-id=<run-identifier>

<trainer-config-file> is the file path of the trainer configuration YAML.
<env_name>(Optional) is the name (including path) of your Unity executable containing the agents to be trained. If <env_name> is not passed, the training will happen in the Editor.
<run-identifier> is a unique name you can use to identify the results of your training runs.

https://github.com/Unity-Technologies/ml-agents/blob/main/docs/Training-ML-Agents.md
"""
def runMLagents(configPath: str, behaviorName: str):
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder = "results/Train/" + behaviorName + "-" + now
    os.makedirs(folder)

    behaviorNameParameter = "--run-id=" + behaviorName + "-" + now
    print("mlagents-learn %s %s --force" % (configPath, behaviorNameParameter))

    # Cliff uncomment and test, check if the command I output to std out above is correct
    # subprocess.run(["mlagents-learn", configPath, behaviorNameParameter, "--force"])