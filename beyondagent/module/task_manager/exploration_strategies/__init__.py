import abc

from beyondagent.schema.task import Task, TaskObjective

class ExploreStrategy(abc.ABC):
    @abc.abstractmethod
    def explore(self, task: Task) -> list[TaskObjective]: ...