from typing import TypedDict

class Person(TypedDict):
    name: str
    age: int
    email: str

new_person1: Person = {
    "name": "Alice",
    "age": 30,
    "email": "xyg@gmail.com"
}
new_person2: Person = {
    "name": "Alice",
    "age": '30', #even if u put string it will not give error because TypedDict does not enforce type checking at runtime
    "email": "xyg@gmail.com"
}

print(type(new_person1))
print(new_person1)
print(new_person1["age"])
print(type(new_person1["age"]))

print(type(new_person2))
print(new_person2)
print(new_person2["age"])
print(type(new_person2["age"]))