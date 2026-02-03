# utils/wrappers.py
class SVMWrapper:
    def __init__(self, pipeline):
        self.pipeline = pipeline
        
    def predict(self, text):
        return self.pipeline.predict([text])[0]
    
    def decision_function(self, text):
        return self.pipeline.decision_function([text])

class LRWrapper:
    def __init__(self, pipeline):
        self.pipeline = pipeline
        
    def predict(self, text):
        return self.pipeline.predict([text])[0]
    
    def decision_function(self, text):
        # If LR supports decision_function, else use predict_proba
        if hasattr(self.pipeline, 'decision_function'):
            return self.pipeline.decision_function([text])
        elif hasattr(self.pipeline, 'predict_proba'):
            # return margin-like score as difference between class probs
            probs = self.pipeline.predict_proba([text])
            return probs[:, 1] - probs[:, 0]
        else:
            raise AttributeError("No decision_function or predict_proba available")
