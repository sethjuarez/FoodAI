from azureml.train.dnn import PyTorch
from azureml.core.environment import CondaDependencies
from azureml.core import ScriptRunConfig, Experiment, Environment, Workspace, RunConfiguration

def dependencies():
    conda_dep = CondaDependencies()
    conda_dep.add_conda_package("matplotlib")
    conda_dep.add_pip_package("numpy")
    conda_dep.add_pip_package("pillow")
    conda_dep.add_pip_package("requests")
    conda_dep.add_pip_package("torchvision")
    conda_dep.add_pip_package("onnxruntime")
    conda_dep.add_pip_package("azureml-defaults")
    return conda_dep

def create_env(is_local=True):
    # environment
    env = Environment(name="foodai-pytorch")
    env.python.conda_dependencies = dependencies()

    if is_local:
        # local docker settings
        env.docker.enabled = True
        env.docker.shared_volumes = True
        env.docker.arguments = [
            "-v", "C:\\projects\\FoodAI\\data:/data"
        ]
    return env

def main():
    # what to run
    script = ScriptRunConfig(source_directory=".", 
                             script="train.py", 
                             arguments=[
                                 "-d", "/data", 
                                 "-e", "10"])
    
    # running the script
    config = RunConfiguration()
    
    # tie everything together
    config.environment = create_env()
    config.target = "local"

    script.run_config = config
    #print(script.run_config)

    # run experiment locally but log to AML
    #ws = Workspace(subscription_id=subscription_id, resource_group=resource_group, workspace_name=workspace_name)
    #ws.write_config()
    ws = Workspace.from_config()
    exp = Experiment(workspace=ws, name="foodai")
    run = exp.submit(config=script)
    run.tag("runtype", "local")
    run.wait_for_completion(show_output=True)

if __name__ == '__main__':
    main()