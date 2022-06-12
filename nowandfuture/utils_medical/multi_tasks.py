import multiprocessing as mp
from logging import warning
from typing import Any, List, Union

#  todo

class Task:
    def __init__(self, _id, _idx=0):
        self._id = _id
        self.idx = _idx

    def do(self, *arg, **kwargs) -> Any:
        raise NotImplementedError()


class Group(Task):
    def __init__(self, _id, tasks: Union[Task, List[Task]]):
        super().__init__(_id, _idx=0)
        if isinstance(tasks, Task):
            tasks = [tasks]

        self.tasks = tasks


    def do(self, params:list) -> Any:
        for task, param in zip(self.tasks, params):
            task.do(*param)


def group_tasks(tasks: List[Task], n_proc) -> List[Group]:
    task_count = len(tasks)
    if n_proc > task_count:
        warning("Tasks is less than processes. Some processes may do nothing.")
        groups = []

        for i in range(n_proc):
            groups.append(Group(i, tasks[i]))
