from typing import TypedDict

class person(TypedDict):
    name = str
    age = int

new_person = person = {'name' : 'Varad', 'age' : '24'}
print(new_person)
print(new_person['name'])



