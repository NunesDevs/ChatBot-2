"""Intent inference never trains, downloads data or loads legacy pickle files."""
import math
import random
from intents_core import MAX_MESSAGE, match_intent, text

UNKNOWN = 'Desculpe, não entendi. Tente uma das intenções cadastradas ou peça ajuda ao operador.'


class ChatService:
    def __init__(self, store, predictor=None, choose=random.choice):
        self.store = store
        self.predictor = predictor
        self.choose = choose

    def state(self):
        dataset = self.store.snapshot()
        ready = self.predictor is not None and self.predictor.source_digest == dataset.digest
        return {'mode': 'neural' if ready else 'patterns', 'intent_count': len(dataset.intents), 'model_stale': self.predictor is not None and not ready}

    def respond(self, message):
        message = text(message, MAX_MESSAGE, 'Mensagem')
        dataset = self.store.snapshot()
        prediction = match_intent(message, dataset)
        if prediction is None and self.predictor is not None and self.predictor.source_digest == dataset.digest:
            prediction = self.predictor.predict(message)
            if prediction is not None:
                confidence = prediction.get('confidence')
                if isinstance(confidence, bool) or not isinstance(confidence, (float, int)) or not math.isfinite(confidence) or not 0.65 <= confidence <= 1:
                    prediction = None
                else:
                    prediction = {**prediction, 'mode': 'neural'}
        if prediction is None:
            return {'response': UNKNOWN, 'intent': None, 'confidence': 0.0, 'mode': 'fallback'}
        intent = next((item for item in dataset.intents if item.tag == prediction['intent']), None)
        if intent is None:
            raise RuntimeError('Modelo incompatível com as intenções cadastradas')
        return {'response': self.choose(intent.responses), **prediction}
