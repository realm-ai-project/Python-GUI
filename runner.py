from datetime import datetime
import os
import subprocess

def runTunerAndMlAgents( configPath: str, envPath: str, behaviorName: str):
    folder = "results/Build" + "-" + behaviorName + "-" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(folder)

    configPathParameter = "--config-path=" + configPath
    envPathParameter = "--env-path" + envPath
    behaviorNameParameter = "--behavior-name=" + behaviorName
    print("realm-tune %s %s %s" % (configPathParameter, envPathParameter, behaviorNameParameter))

    # Ken uncomment and test, check if the command I output to std out above is correct
    # subprocess.run(["realm-tune", configPathParameter, envPathParameter, behaviorNameParameter])


def runMLagents():
    print("mlagents")