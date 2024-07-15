import importlib.util
from typing import TYPE_CHECKING

from .utils import BestRun


if TYPE_CHECKING:
    from .trainer import Trainer


def is_optuna_available() -> bool:
    return importlib.util.find_spec("optuna") is not None


def default_hp_search_backend():
    if is_optuna_available():
        return "optuna"

def run_hp_search_optuna(trainer, n_trials, direction, **kwargs):

    def _objective(trial, checkpoint_dir=None):
        print("objectify")
        checkpoint = None
        if checkpoint_dir:
            for subdir in os.listdir(checkpoint_dir):
                if subdir.startswith(PREFIX_CHECKPOINT_DIR):
                    checkpoint = os.path.join(checkpoint_dir, subdir)
        if not checkpoint:
            # free GPU memory
            del trainer.model
            gc.collect()
            torch.cuda.empty_cache()
        trainer.objective = None
        trainer.train(resume_from_checkpoint=checkpoint, trial=trial)
        # If there hasn't been any evaluation during the training loop.
        if getattr(trainer, "objective", None) is None:
            metrics = trainer.evaluate()
            trainer.objective = trainer.compute_objective(metrics)
        return trainer.objective

    timeout = kwargs.pop("timeout", None)
    n_jobs = kwargs.pop("n_jobs", 1)
    study = optuna.create_study(direction=direction, **kwargs)
    study.optimize(_objective, n_trials=n_trials,
                   timeout=timeout, n_jobs=n_jobs)
    best_trial = study.best_trial
    return BestRun(str(best_trial.number), best_trial.value, best_trial.params)
