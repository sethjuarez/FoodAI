from azureml.core import Workspace

ws = Workspace.from_config()
print(ws.webservices['foodai'].get_logs())