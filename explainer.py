import xgboost
import tempfile
from joblib import dump, load
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import shap
from matplotlib import pyplot as plt
from typing import Dict


class TemporaryDirectory(object):
    """Create and return a temporary directory.  This has the same
    behavior as mkdtemp but can be used as a context manager.  For
    example:
        with TemporaryDirectory() as tmpdir:
            ...
    Upon exiting the context, the directory and everything contained
    in it are removed.
    """

    def __init__(self, suffix=None, prefix=None, dir=None):
        self.name = mkdtemp(suffix, prefix, dir)
        self._finalizer = _weakref.finalize(
            self, self._cleanup, self.name,
            warn_message="Implicitly cleaning up {!r}".format(self))

    @classmethod
    def _cleanup(cls, name, warn_message):
        _shutil.rmtree(name)
        _warnings.warn(warn_message, ResourceWarning)

    def __repr__(self):
        return "<{} {!r}>".format(self.__class__.__name__, self.name)

    def __enter__(self):
        return self.name

    def __exit__(self, exc, value, tb):
        self.cleanup()

    def cleanup(self):
        if self._finalizer.detach():
            _shutil.rmtree(self.name)


class ModelPipeline:
    def __init__(self, X, y, label_encoding:bool=False) -> None:
        self.X = X
        if label_encoding:
            self.y = LabelEncoder().fit_transform(y)
        else:
            self.y = y
        self.y = self.y.astype(int)
        print(f"X.shape is {X.shape}")
        print(f"y.shape is {y.shape}")
        print("Model pipeline object created...")

    @staticmethod
    def silent_rm(filename) -> None:
        """silent_rm.
        :param filename:
        """
        try:
            os.remove(filename)
        except OSError as e:
            if e.errno != errno.ENOENT:
                # re-raise exception if a different error occurred
                raise

    @staticmethod
    def plot_PRCurve(model, y_score, y_test) -> None:
        # Plot precision-recall curve on test set
        precision = dict()
        recall = dict()
        for num, one_class in enumerate(model.classes_):
            print(f"[Plotting PR curve for given class: {one_class}]")
            precision[num], recall[num], _ = precision_recall_curve(
                    y_test.loc[:, one_class], y_score[:, num])
            plt.plot(recall[num], precision[num], lw=2, label='class {}'.format(num))
        plt.xlabel("recall")
        plt.ylabel("precision")
        plt.title("precision vs. recall curve")
        plt.show()

    @staticmethod
    def plot_ROCCurve(model, y_score, y_test) -> None:
        # Plot RCO curve on the test set
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for num, one_class in enumerate(model.classes_):
            print(f"Plotting ROC curve for given class: {one_class}")
            fpr[num], tpr[num], _ = roc_curve(y_test.loc[:, one_class], y_score[:, num])
            roc_auc[num] = auc(fpr[num], tpr[num])

        plt.figure()
        for num in range(len(model.classes_)):
            plt.plot(fpr[num], tpr[num], label='ROC curve (area = %0.2f)' % roc_auc[num])
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.show()

    def grid_search(self, params:Dict[str, float], random:bool=False) -> None:
        X, y = self.X, self.y
        clf = xgboost.XGBClassifier(objective='multi:softmax', random_state=42)
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        if random:
            grid = RandomizedSearchCV(clf, params, cv=kfold, n_iter=20, n_jobs=-1, random_state=2)
        else:
            grid = GridSearchCV(clf, params, cv=kfold, n_jobs=-1)
        # Fit grid_reg on X_train and y_train
        grid.fit(X, y, eval_set=[(X, y)], eval_metric='mlogloss')
        # Extract best params
        best_params = grid.best_params_
        # Print best params
        print("Best params:", best_params)
        # Compute best score
        best_score = grid.best_score_
        # Print best score
        print("Best score: {:.5f}".format(best_score))
        return grid

    def training_and_eval(self, tmpdir: str, use_pickle: bool) -> None:
        """Basic training continuation."""
        # Train 128 iterations in 1 session
        X, y = self.X, self.y
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2)
        clf = xgboost.XGBClassifier(objective='multi:softmax', 
                random_state=42, 
                n_estimators=128, 
                use_label_encoder=False)
        clf.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric="mlogloss")
        print("Total boosted rounds:", clf.get_booster().num_boosted_rounds())
        # make predictions for test data
        y_pred = clf.predict(X_test)
        # evaluate predictions
        accuracy = accuracy_score(y_test, y_pred)
        print("Test Accuracy: %.2f%%" % (accuracy * 100.0))
        _ = self.plot_various_features(clf)

    def plot_various_features(self, model):
        xgboost.plot_importance(model)
        plt.title("xgboost.plot_importance(model)")
        plt.tight_layout()
        plt.show()

        xgboost.plot_importance(model, importance_type="cover")
        plt.title('xgboost.plot_importance(model, importance_type="cover")')
        plt.tight_layout()
        plt.show()

        xgboost.plot_importance(model, importance_type="gain")
        plt.title('xgboost.plot_importance(model, importance_type="gain")')
        plt.tight_layout()
        plt.show()

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(self.X)
        shap.summary_plot(shap_values, self.X)
