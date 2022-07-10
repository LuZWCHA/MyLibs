import multiprocessing as mp
from logging import warning
from typing import Any, List, Union


class Task:
    def __init__(self, _id=-1, _idx=0):
        self._id = _id
        self.idx = _idx

    def do(self, *arg, **kwargs) -> Any:
        raise NotImplementedError()

    def __str__(self):
        return f"Group ID: {self._id}. Index: {self.idx}."


class Group(Task):
    def __init__(self, _id, tasks: Union[Task, List[Task]]):
        super().__init__(_id, _idx=0)
        if isinstance(tasks, Task):
            tasks._id = _id
            tasks = [tasks]

        self.tasks = tasks
        for idx, task in enumerate(self.tasks):
            task._id = _id
            task._idx = idx

    def do_all(self, params: list) -> Any:
        for task, param in zip(self.tasks, params):
            task.do(*param)

    def do(self, *arg, **kwargs) -> Any:
        self.do_all(*arg, **kwargs)

    def __len__(self):
        return len(self.tasks)


def group_tasks(tasks: List[Task], n_proc) -> List[Group]:
    task_count = len(tasks)
    group_size = task_count // n_proc
    groups = []

    if n_proc > task_count:
        warning("Tasks is less than process number. Some processes may do nothing.")

        for i in range(n_proc):
            groups.append(Group(i, tasks[i]))
    else:
        for i in range(n_proc):
            groups.append(Group(i, tasks[i * group_size: min((i + 1) * group_size, task_count)]))

    return groups


def do_tasks(tasks: Union[List[Task], Task], n_proc):
    pass
