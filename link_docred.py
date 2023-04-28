import json
import random
import numpy as np 
dataset = json.load(open('/data/lyt/workspace/RollingKoRC/rolling_docred.json'))

data_to_link = []
for data in dataset:
    related_entities = set(sum([
        [triple['head_entity'],triple['tail_entity']] for triple in data['re_result']
    ],[]))
    for v_idx,vertex_mentions in enumerate(data['vertexSet']):
        mention_name_list = [mention['name'] for mention in vertex_mentions]         
        if len(set(mention_name_list).intersection(related_entities)) > 0:
            if np.mean([len(mention) for mention in mention_name_list]) <= 2 and \
                'AI' not in mention_name_list:
                selected_mention = random.choice(vertex_mentions)
                sent_id = selected_mention['sent_id']
                pos_start = max(0,selected_mention['pos'][0])
                pos_end = min(len(data['sents'][sent_id]),selected_mention['pos'][1])
                context_left = ''
                context_right = ''
                selected_name = " ".join(data['sents'][sent_id][pos_start:pos_end])
                for idx,sent in enumerate(data['sents']):
                    if idx < sent_id:
                        context_left += " ".join(sent)
                    elif idx == sent_id:
                        context_left += " ".join(sent[:pos_start])
                        context_right += " ".join(sent[pos_end:])
                    else:
                        context_right += " ".join(sent)
            else:
                selected_name = random.choice(mention_name_list)
                pos = data['content'].find(selected_name)
                context_left = data['content'][:pos]
                context_right = data['content'][pos+len(selected_name):]
            data_to_link.append(
                {
                    "id": data['docId'] + '_' + str(v_idx),
                    "label":'',
                    "label_id": -1,
                    "context_left": context_left,
                    "mention": selected_name,
                    "context_right": context_right,
                }
            )            


import blink.main_dense as main_dense
import argparse

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

_, _, _, _, _, predictions, scores = main_dense.run(args, None, *models, test_data=data_to_link[:])

from prettytable import PrettyTable

x = PrettyTable()
x.field_names = ['id','mention','context','pred','score']


x.add_rows(sorted([
    [
        data['id'],
        data['mention'],
        data['context_left'][-20:] + "\u0332".join(" " +data['mention'] + " ") + data['context_right'][:20],
        pred[0],
        float(score[0])
    ]
    for data,pred,score in zip(data_to_link,predictions,scores)
],key=lambda x:x[3]))

with open('/data/lyt/workspace/RollingKoRC/rolling_docred_linked_result.json','w') as f:
    f.write(x.get_json_string(indent=4,ensure_ascii=False))
