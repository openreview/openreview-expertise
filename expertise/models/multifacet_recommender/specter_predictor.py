from typing import List, Dict

from allennlp.common import JsonDict
from allennlp.data import Instance
from allennlp.predictors import Predictor
from overrides import overrides
from specter.predictor import SpecterPredictor


@Predictor.register('specter_predictor_custom')
class SpecterPredictorCustom(SpecterPredictor):

    @overrides
    def predict_json(self, paper: JsonDict) -> Dict:
        ret = {}
        for key in ['paper_id', 'title', 'abstract', 'authors', 'venue']:
            try:
                ret[key] = paper[key]
            except KeyError:
                pass
        ret['embedding'] = []
        try:
            if hasattr(self._model, 'bert_finetune') and self._model.bert_finetune:
                if not (paper['title'] or paper['abstract']):
                    return ret
            else:
                if not (paper['title'] or paper['abstract']):
                    return ret
        except KeyError:
            return ret

        self._dataset_reader.text_to_instance(paper)

        instance = self._dataset_reader.text_to_instance(paper)

        outputs = self._model.forward_on_instance(instance)

        ret['embedding'] = outputs['embedding'].tolist()
        return ret

    @overrides
    def predict_batch_json(self, inputs: List[JsonDict]) -> List[JsonDict]:
        instances = []
        skipped_idx = []
        for idx, json_dict in enumerate(inputs):
            paper = {}
            if 'title' not in json_dict:
                skipped_idx.append(idx)
                continue
            skip = False
            for key in ['paper_id', 'title', 'abstract', 'authors', 'venue']:
                try:
                    paper[key] = json_dict[key]
                except KeyError:
                    pass
            paper['embedding'] = []
            try:
                if hasattr(self._model, 'bert_finetune') and self._model.bert_finetune:
                    # this model concatenates title/abstract
                    if not (json_dict['title'] or json_dict['abstract']):
                        skip = True
                else:
                    # one of title and abstract must be present
                    if not (json_dict['title'] or json_dict['abstract']):
                        skip = True

            except KeyError:
                skip = True
            if not skip:
                instances.append(self._dataset_reader.text_to_instance(json_dict))
            else:
                skipped_idx.append(idx)
        if instances:
            outputs = self._model.forward_on_instances(instances)
        else:
            outputs = []
        k = 0
        results = []
        for j in range(len(inputs)):
            paper = {}
            for key in ['paper_id', 'title']:
                try:
                    paper[key] = inputs[j][key]
                except KeyError:
                    pass
            paper['embedding'] = []
            if not skipped_idx or k >= len(skipped_idx) or skipped_idx[k] != j:
                paper['embedding'] = outputs[j - k]['embedding'].tolist()
                results.append(paper)
            else:
                paper['embedding'] = []
                results.append(paper)
                k += 1
        return results