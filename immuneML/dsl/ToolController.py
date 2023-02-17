from prefect import flow, task, context
import subprocess
from immuneML.dsl.symbol_table.SymbolType import SymbolType
from immuneML.dsl.symbol_table.SymbolTable import SymbolTable


# run tool creator
@flow
def get_dataset(symbol_table, workflow_specification):
    dataset_path = run_dataset_tool(symbol_table.get_by_type(SymbolType.TOOL)[1].item["path"])
    print("Path to dataset created by the tool: ", dataset_path)
    workflow_specification["definitions"]["datasets"]["my_dataset"]["params"]["path"] = dataset_path["path"]
    workflow_specification["definitions"]["datasets"]["my_dataset"]["params"]["metadata_file"] = dataset_path[
        "metadata"]

    return workflow_specification


# simple implementation of running tool as a subprocess. Tool returns path to created dataset
@task
def run_dataset_tool(tool_path):
    # run tool as a subprocess. Create dataset tool and add to the folder specified
    p = subprocess.Popen(["python", tool_path, "/Users/oskar/Desktop/Dataset_tool"], stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)
    # out, error = p.communicate()
    # print("in tool parser", out.decode("ascii"))

    return {"path": "/Users/oskar/Desktop/Dataset_tool/quickstart_data/repertoires",
            "metadata": "/Users/oskar/Desktop/Dataset_tool/quickstart_data/metadata.csv"}
