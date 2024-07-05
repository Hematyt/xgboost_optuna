import optuna
import dataset
import sklearn.datasets
import sklearn.ensemble
import sklearn.model_selection
import sklearn.svm

X_train, X_val, y_train, y_val = dataset.create_dataset()


def objective(trial):

    max_depth = trial.suggest_int("max_depth", 2, 50, log=True)
    n_estimators = trial.suggest_int("n_estimators", 50, 500, log=True)
    max_features = trial.suggest_categorical("max_features", ['sqrt', 'log2', None])
    class_weight = trial.suggest_categorical('class_weight', ['balanced', 'balanced_subsample'])

    classifier_obj = sklearn.ensemble.RandomForestClassifier(
        max_depth=max_depth,
        n_estimators=n_estimators,
        max_features=max_features,
        class_weight=class_weight
    )

    score = sklearn.model_selection.cross_val_score(classifier_obj, X_train, y_train, n_jobs=-1, cv=3)
    accuracy = score.mean()
    return accuracy


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)
    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
