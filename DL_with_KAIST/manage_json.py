import json

data = {
    "olivia" : {
        "gender": "female",
        "age" : 25,
        "hobby" : ["reading", "music"]
    },
    "Tyler" : {
        "gender": "male",
        "age" : 28,
        "hobby" : ["development", "painting"]
    }
}

file_path = "./test.json"

with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(data, f)

