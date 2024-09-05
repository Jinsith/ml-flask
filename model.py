from sklearn.linear_model import LinearRegression
import joblib

class HousePriceModel:
    def __init__(self):
        self.model = LinearRegression()

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def save(self, filename):
        joblib.dump(self.model, filename)

    @classmethod
    def load(cls, filename):
        model = cls()
        model.model = joblib.load(filename)
        return model