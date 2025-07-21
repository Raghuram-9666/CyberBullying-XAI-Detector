# utils/wrappers.py
class SVMWrapper:
    def __init__(self, pipeline):
        self.pipeline = pipeline
        
    def predict(self, text):
        return self.pipeline.predict([text])[0]

class LRWrapper:
    def __init__(self, pipeline):
        self.pipeline = pipeline
        
    def predict(self, text):
        return self.pipeline.predict([text])[0]