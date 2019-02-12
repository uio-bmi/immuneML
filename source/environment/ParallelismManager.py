import warnings


class ParallelismManager:
    """
    Includes a dict of all allocated resources per job type: using that information, it assigns
    resources for new jobs in order to balance parallelism

    The classes which ask for resources are obliged to free the resources (let the manager know that they have finished)
    """
    resources = {"unidentified": []}
    total_cores = 4

    @staticmethod
    def assign_cores_to_job(step_type=None):
        if step_type is None:
            ParallelismManager.resources["unidentified"].append(2)
            cores = 2
        elif step_type in ParallelismManager.resources:
            ParallelismManager.resources[step_type].append(2)
            cores = 2
        else:
            ParallelismManager.resources[step_type] = [2]
            cores = 2
        return cores

    @staticmethod
    def free_cores(step_type = None, cores: int = 2):
        if step_type is None:
            ParallelismManager.resources["unidentified"].remove(cores)
        elif step_type in ParallelismManager.resources:
            ParallelismManager.resources[step_type].remove(cores)
        else:
            warnings.warn("ParallelismManager: resources not freed; no such step_type was previously noted.")
