"""Controller-side step accounting; acknowledgements follow optimizer completion."""

from dataclasses import replace

from .contracts import ClusterState, WorkerIdentity, _integer


class StepCommit:
    def __init__(self, state: ClusterState):
        self._state = state
        self._acknowledged: set[str] = set()

    @property
    def state(self) -> ClusterState:
        return self._state

    def acknowledge(self, worker: WorkerIdentity, completed_global_step: int) -> bool:
        """Return True only when every current participant confirms this step."""
        _integer("completed_global_step", completed_global_step)
        if worker not in self._state.workers:
            raise ValueError("worker identity must match the current generation and rank")
        if completed_global_step != self._state.committed_global_step + 1:
            raise ValueError("acknowledgement must be for the next global step")
        if worker.worker_id in self._acknowledged:
            raise ValueError("worker already acknowledged this global step")
        self._acknowledged.add(worker.worker_id)
        if len(self._acknowledged) != len(self._state.workers):
            return False
        self._state = replace(self._state, committed_global_step=completed_global_step)
        self._acknowledged.clear()
        return True
