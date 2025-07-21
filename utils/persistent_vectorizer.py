import types
import emoji
import re
from sklearn.feature_extraction.text import TfidfVectorizer

class PersistentVectorizer(TfidfVectorizer):
    def __getstate__(self):
        state = super().__getstate__()
        if callable(state.get('tokenizer')):
            state['tokenizer_code'] = state['tokenizer'].__code__
            state['tokenizer_globals'] = {
                'emoji': emoji,
                're': re
            }
            del state['tokenizer']
        return state

    def __setstate__(self, state):
        if 'tokenizer_code' in state:
            state['tokenizer'] = types.FunctionType(
                state['tokenizer_code'],
                state['tokenizer_globals'],
                name='custom_tokenizer'
            )
        super().__setstate__(state)