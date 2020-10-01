from azureml.core import Workspace

ws = Workspace.from_config()
print(ws.webservices['bork-svc'].get_logs())