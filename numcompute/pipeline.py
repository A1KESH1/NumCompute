class Pipeline:
    def __init__(self, steps):
        if not isinstance(steps, list) or not steps:
            raise ValueError("steps must be a non-empty list.")
        self.steps = list(steps)
        self._is_fitted = False
        self._validate_steps()

    def _validate_steps(self):
        if not self.steps:
            raise ValueError("steps must be non-empty.")
        names = set()
        for i, item in enumerate(self.steps):
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                raise ValueError("Each step must be a (name, object) pair.")
            name, step = item
            if not isinstance(name, str) or not name.strip() or name in names:
                raise ValueError("Step names must be non-empty, unique strings.")
            names.add(name)
            if not callable(getattr(step, "fit", None)):
                raise ValueError(f"Step '{name}' must implement fit.")
            if not callable(getattr(step, "transform", None)):
                if i != len(self.steps) - 1 or not callable(getattr(step, "predict", None)):
                    raise ValueError(f"Step '{name}' must implement transform.")

    def _check_fitted(self):
        if not self._is_fitted:
            raise ValueError("Pipeline must be fitted before transform or predict.")
        self._validate_steps()

    def _require_transform(self):
        if not callable(getattr(self.steps[-1][1], "transform", None)):
            raise ValueError("The final estimator cannot transform; use fit and predict.")

    @staticmethod
    def _fit_transform_step(step, X):
        if callable(getattr(step, "fit_transform", None)):
            return step.fit_transform(X)
        step.fit(X)
        return step.transform(X)

    def _fit(self, X, y, return_transformed):
        self._is_fitted = False
        self._validate_steps()
        if return_transformed:
            self._require_transform()
        Xt = X
        for _, step in self.steps[:-1]:
            Xt = self._fit_transform_step(step, Xt)
        final = self.steps[-1][1]
        if callable(getattr(final, "transform", None)):
            if return_transformed:
                Xt = self._fit_transform_step(final, Xt)
            else:
                final.fit(Xt)
        elif y is None:
            final.fit(Xt)
        else:
            final.fit(Xt, y)
        self._is_fitted = True
        return Xt if return_transformed else self

    def fit(self, X, y=None):
        return self._fit(X, y, return_transformed=False)

    def transform(self, X):
        self._check_fitted()
        self._require_transform()
        Xt = X
        for _, step in self.steps:
            Xt = step.transform(Xt)
        return Xt

    def fit_transform(self, X, y=None):
        return self._fit(X, y, return_transformed=True)

    def predict(self, X):
        self._check_fitted()
        final = self.steps[-1][1]
        if not callable(getattr(final, "predict", None)):
            raise ValueError("The final step must implement predict.")
        Xt = X
        for _, step in self.steps[:-1]:
            Xt = step.transform(Xt)
        return final.predict(Xt)
