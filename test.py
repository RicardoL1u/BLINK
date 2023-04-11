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

data_to_link = [ 
    {
        "id": 0,
        "label": "unknown",
        "label_id": -1,
        "context_left": "England’s Euro 2024 qualifying campaign could barely have got off to a better start, with six points claimed from their opening two games against ".lower(),
        "mention": "Italy".lower(),
        "context_right": " and Ukraine".lower(),
    },
    {
        "id": 1,
        "label": "unknown",
        "label_id": -1,
        "context_left": "".lower(),
        "mention": "England".lower(),
        "context_right": "'s Euro 2024 qualifying campaign could barely have got off to a better start, with six points claimed from their opening two games against Italy and Ukraine".lower(),
    },
    {
        "id": 2,
        "label": "unknown",
        "label_id": -1,
        "context_left": "But surely the biggest surprise is".lower(),
        "mention": "Ronaldo".lower(),
        "context_right": "s drop in value, despite his impressive record of 53 goals and 14 assists in 75 appearances for Juventus".lower(),
    },
    {
        "id": 3,
        "label": "unknown",
        "label_id": -1,
        "context_left": "Qualifying for Euro 2024 presents the Three Lions with a chance to move on from their ".lower(),
        "mention": "Qatar".lower(),
        "context_right": " disappointment as they aim to win their first major trophy since 1966".lower(),
    },
    {
        "id": 4,
        "label": "unknown",
        "label_id": -1,
        "context_left": "Qualifying for World Cup 2022 presents the Three Lions with a chance to move on from their ".lower(),
        "mention": "Wembley".lower(),
        "context_right": " disappointment as they aim to win their first major trophy since 1966".lower(),
    },
    {
        "id": 5,
        "label": "unknown",
        "label_id": -1,
        "context_left": "In the Spanish national derby in the 2015-16 season, ".lower(),
        "mention": "the Argentine No.10".lower(),
        "context_right": " from the Barcelona team missed a penalty against the Spanish national goal Casillas.".lower(),
    },
    {
        "id": 6,
        "label": "unknown",
        "label_id": -1,
        "context_left": "In 1987, ".lower(),
        "mention": "Chiang Kai-shek's son ".lower(),
        "context_right": "was successfully elected president of the Republic of China".lower(),
    },
    {
        "id": 7,
        "label": "unknown",
        "label_id": -1,
        "context_left": "In 2028, the opening ceremony of the Olympic Games was successfully held in ".lower(),
        "mention": "🇧🇪".lower(),
        "context_right": "".lower(),
    },
    

]

_, _, _, _, _, predictions, scores, = main_dense.run(args, None, *models, test_data=data_to_link)

for pred in predictions:
    print(pred[:10])