from pathlib import Path

from torchtext.data import Dataset, Example, Field, ReversibleField, LabelField
from tqdm.auto import tqdm
import ipdb

from src.preprocess import preprocess

SLOT_TYPES = ['domain', 'slot', 'gate', 'val', 'fertility']


class Example(Example):
    @classmethod
    def fromchuck(cls, data, history_fields, slot_fields):
        ex = cls()
        data = data.strip().split('\n')
        assert len(data) == 32

        for (name, field), val in zip(history_fields, data[:2]):
            setattr(ex, name, field.preprocess(val))

        for vals in data[2:]:
            vals = vals.strip().split('\t')
            for val, field, slot in zip(vals, slot_fields, SLOT_TYPES):
                name = f"{vals[0]}_{vals[1]}_{slot}"
                if field is not None:
                    setattr(ex, name, field.preprocess(val))
        return ex


class MultiWozDSTDataset(Dataset):
    urls = ['http://140.112.29.239:8000/data2.1.tgz']
    dirname = 'data2.1'
    name = 'MultiWoz_2.1_NADST_Version'

    def __init__(self, 
                 path, 
                 history_fields, 
                 slot_fields, 
                 **kwargs):
        fields = history_fields + slot_fields
        chucks = path.read_text().strip().split('\n\n')
        examples = [Example.fromchuck(chuck, history_fields, slot_fields) 
                    for chuck in tqdm(chucks)]
        super(MultiWozDSTDataset, self).__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls,
               history_fields, 
               slot_fields,
               root=".data",
               train="nadst_train_dials.json",
               validation="nadst_dev_dials.json",
               test="nadst_test_dials.json",
               **kwargs):
        assert len(history_fields) == 2
        assert len(slot_fields) == 5
        cls.download(root)
        dirname = Path(root) / cls.name / cls.dirname 
        train = dirname / train
        validation = dirname / validation
        test = dirname / test
        ontology_path = dirname / 'multi-woz/MULTIWOZ2.1/ontology.json'

        train_processed = dirname / 'train_processed.txt'
        validation_processed = dirname / 'validation_processed.txt'
        test_processed = dirname / 'test_processed.txt'

        preprocess(train, ontology_path, train_processed)
        preprocess(validation, ontology_path, validation_processed)
        preprocess(test, ontology_path, test_processed)

        train_data = cls(train_processed, history_fields, slot_fields)
        val_data = cls(validation_processed, history_fields, slot_fields)
        test_data = cls(test_processed, history_fields, slot_fields)
        return tuple(d for d in (train_data, val_data, test_data)
                     if d is not None)


if __name__ == '__main__':
    text_field = ReversibleField()
    gate_field = LabelField()
    fertility_field = LabelField()
    history_fields = [('history', text_field), ('history_delex', text_field)]
    slot_fields = [('attraction_area_domain', text_field), text_field, gate_field, text_field, fertility_field]
    train, val, test = MultiWozDSTDataset.splits(history_fields, slot_fields)
    ipdb.set_trace()
