from pet.utils import InputExample

from pet.pvp import (
    PVP, FilledPattern,
)

from pet.utils import get_verbalization_ids


class CustomCopaPVP(PVP):
    def get_parts(self, example: InputExample) -> FilledPattern:
        premise = self.remove_final_punc(self.shortenable(example.text_a))
        choice1 = self.remove_final_punc(self.lowercase_first(example.meta['choice1']))
        choice2 = self.remove_final_punc(self.lowercase_first(example.meta['choice2']))

        question = example.meta['question']
        assert question in ['cause', 'effect']

        example.meta['choice1'], example.meta['choice2'] = choice1, choice2
        num_masks = max(len(get_verbalization_ids(c, self.wrapper.tokenizer, False)) for c in [choice1, choice2])

        if question == 'cause':
            if self.pattern_id == 0:
                raise NotImplementedError

        else:
            if self.pattern_id == 0:
                raise NotImplementedError
