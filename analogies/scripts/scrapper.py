import requests
import json

# ID for critical role wiki
CR_ID = "1163770"

# Request the list of critical role npcs
payload = {"expand": "1","category": "Non-player_characters", 'limit': "700"}
r = requests.get("https://criticalrole.fandom.com/api/v1/Articles/List", params=payload)


# Filter out unwanted pages (category pages mostly) and discard unneed data
npcs = [{'title': s['title'], 'url': s['url'], 'id': s['id']} for s in r.json()['items'] if not (('Category:' in s['url']) or ('Supporting Characters' in s['title']))]

# create our nodes
nodes = [{'name': s['title']} for s in npcs]
npc_string = json.dumps({'nodes': nodes}, indent=4)
file_object = open("../data/nodes.json", "w+")
file_object.write(npc_string)
file_object.close()

# payload = {'id': npcs[0]['id']}
# r = requests.get("https://criticalrole.fandom.com/api/v1/Articles/AsSimpleJson/", params=payload)
# print(r.url)
# print(json.dumps(r.json(), indent=4))