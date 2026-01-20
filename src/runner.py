from typing import List, Dict, Any, Optional
from tqdm import tqdm

from .tasks.base import BaseTask
from .methods.base import BaseMethod

class ExperimentRunner:
    """
    Runs analysis methods across multiple tasks and saves results.

    Example usage:
        runner = ExperimentRunner()
        runner.add_task(BlackmailTask.load())
        runner.add_task(ScruplesTask.load())
        runner.add_method(LLMMonitor(model="claude-3-5-sonnet"))
        runner.add_method(KeywordDetector())
        results = runner.run()
    """

    def __init__(self):
        self.tasks: List[BaseTask] = []
        self.methods: List[BaseMethod] = []

    def add_task(self, task_name: str) -> "ExperimentRunner":
        """
        Add a task to run methods on.

        Args:
            task: A loaded task instance.

        Returns:
            self for method chaining.
        """
        self.tasks.append(BaseTask(task_name))
        return self

    def add_method(self, method_name: str) -> "ExperimentRunner":
        """
        Add a method to run on all tasks.

        Args:
            method: A method instance.

        Returns:
            self for method chaining.
        """
        self.methods.append(BaseMethod(method_name))
        return self

    def remove_task(self, task_name: str) -> "ExperimentRunner":
        """
        Remove a task by name.

        Args:
            task_name: Name of the task to remove.

        Returns:
            self for method chaining.
        """
        self.tasks = [t for t in self.tasks if t.name != task_name]
        return self

    def remove_method(self, method_name: str) -> "ExperimentRunner":
        """
        Remove a method by name.

        Args:
            method_name: Name of the method to remove.

        Returns:
            self for method chaining.
        """
        self.methods = [m for m in self.methods if m.name != method_name]
        return self

    def run(
        self,
        verbose: bool = True,
        save_results: bool = True,
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Run all methods on all tasks.

        Args:
            verbose: Whether to print progress information.
            save_results: Whether to save predictions to task CSVs.

        Returns:
            Nested dict: {task_name: {method_name: {metric_name: value}}}
        """
        results: Dict[str, Dict[str, Dict[str, float]]] = {}

        for task in self.tasks:
            if verbose:
                print(f"\n{'='*50}")
                print(f"Task: {task.name}")
                print(f"{'='*50}")

            results[task.name] = {}

            # Load all rollouts and ground truth for this task
            rollouts = []
            ground_truth = []

            iterator = list(task.data.iterrows())
            if verbose and tqdm is not None:
                iterator = tqdm(iterator, desc="Loading rollouts", leave=False)

            for idx, row in iterator:
                rollouts.append(task.get_rollout(row))
                ground_truth.append(task.get_ground_truth(row))

            # Run each method
            for method in self.methods:
                if verbose:
                    print(f"\n  Method: {method.name}")

                predictions = method.predict_batch(rollouts)

                metrics = task.evaluate(predictions, ground_truth)
                results[task.name][method.name] = metrics

                if save_results:
                    task.save_results(method.name, predictions)

                if verbose:
                    metrics_str = ", ".join(
                        f"{k}={v:.4f}" for k, v in metrics.items()
                    )
                    print(f"    Results: {metrics_str}")

        return results
