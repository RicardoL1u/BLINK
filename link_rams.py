import nltk
import json
# from transformers import AutoTokenizer
# tokenizer = AutoTokenizer.from_pretrained('gpt2')
from collections import defaultdict
import blink.main_dense as main_dense
import argparse

url2qid = json.load(open('url2qid.json'))
def get_predicted_url(title:str)->str:
    prefix = 'https://en.wikipedia.org/wiki/'
    title = title.replace(' ','_')
    title = title.replace('?', '%3F')
    # title = title.replace('&amp;', '%26')
    # title = title.replace('\'','%27')
    return prefix+title

models_path = "models/" # the path where you stored the BLINK models

config = {
    "test_entities": None,
    "test_mentions": None,
    "interactive": False,
    "top_k": 10,
    "biencoder_model": models_path+"biencoder_wiki_large.bin",
    "biencoder_config": models_path+"biencoder_wiki_large.json",
    "entity_catalogue": models_path+"entity.jsonl",
    "entity_encoding": models_path+"all_entities_large.t7",
    "crossencoder_model": models_path+"crossencoder_wiki_large.bin",
    "crossencoder_config": models_path+"crossencoder_wiki_large.json",
    "fast": False, # set this to be true if speed is a concern
    "output_path": "logs/" # logging directory
}

args = argparse.Namespace(**config)

models = main_dense.load_models(args, logger=None)

stopwords = nltk.corpus.stopwords.words('english')

lines = open('/data/lyt/workspace/FakeEvent/data/RAMS_1.0c/data/train.jsonlines').readlines()
lines.extend(open('/data/lyt/workspace/FakeEvent/data/RAMS_1.0c/data/dev.jsonlines').readlines())
lines.extend(open('/data/lyt/workspace/FakeEvent/data/RAMS_1.0c/data/test.jsonlines').readlines())
dataset = [json.loads(line) for line in lines]
for example in dataset:
    example['tokens'] = sum(example['sentences'],[])


data_to_link = []
for example in dataset:
    for span in example['ent_spans']:
        mention_tokens = example['tokens'][span[0]:span[1]+1]
        if len(mention_tokens) > 15:
            mention_tokens = mention_tokens[:15]
        if " ".join(mention_tokens) not in stopwords:
            data_to_link.append(
                {
                    "id": example['doc_key'] + '_' + str(span[0]),
                    "label": span[2][0][0],
                    "label_id": -1,
                    "context_left": " ".join(example['tokens'][:span[0]]).lower(),
                    "mention": " ".join(mention_tokens).lower(),
                    "context_right": " ".join(example['tokens'][span[1]+1:]).lower(),
                }
            )
            
        

# data_to_link = [
#     {
#         "id": example['doc_key'] + '_' + str(span[0]),
#         "label": span[2][0][0],
#         "label_id": -1,
#         "context_left": " ".join(example['tokens'][:span[0]]).lower(),
#         "mention": " ".join(example['tokens'][span[0]:span[1]+1]).lower(),
#         "context_right": " ".join(example['tokens'][span[1]+1:]).lower(),
#     }
#     for span in example['ent_spans'] for example in dataset if " ".join(example['tokens'][span[0]:span[1]+1]) not in stopwords
# ]

_, _, _, _, _, predictions, scores = main_dense.run(args, None, *models, test_data=data_to_link[:])

args2mention = defaultdict(list)
args2qid = defaultdict(list)
for data,pred,score in zip(data_to_link,predictions,scores):
    if score[0] > 0:
        args2mention[data['label']].append(pred[0])
        if get_predicted_url(pred[0]) in url2qid.keys():
            args2qid[data['label']].append(url2qid[get_predicted_url(pred[0])])
        
with open('/data/lyt/workspace/FakeEvent/data/RAMS_1.0c/data/args2mention.json','w') as f:
    json.dump(args2mention,f,indent=4,ensure_ascii=False)

with open('/data/lyt/workspace/FakeEvent/data/RAMS_1.0c/data/args2qid.json','w') as f:
    json.dump(args2qid,f,indent=4,ensure_ascii=False)

    