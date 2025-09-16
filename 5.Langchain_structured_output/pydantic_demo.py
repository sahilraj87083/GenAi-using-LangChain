from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class Student(BaseModel):
    name: str = "Unknown" #default value
    age: Optional[int] = None
    email : Optional[EmailStr] = None
    cgpa: float = Field(gt=0, lt=10, description="CGPA must be between 0 and 10", default=5) #gt: greater than, lt: less than


# new_student = {"name" : "John Doe" }
# new_student = {"name": 32 } #name in int : it will give error because pydantic enforces type checking at runtime

# new_student = {"age": "32" } #age in string : it will not give error because pydantic will convert it to int using type coercion
# new_student = {"email": "invalidemail" } # it will give error because pydantic checks for valid email format

# new_student = {"name": "Jane Doe", "age": 25, "email": "abc@gmail.com"}

# new_student = {"name": "Jane Doe", "age": 25, "cgpa" : 12.5} # it will give error because cgpa is not between 0 and 10}
new_student = {"name": "Jane Doe", "age": 25, "cgpa" : 5.5} # it will not give error because cgpa is between 0 and 10

student = Student(** new_student) 
print(type(student))
print(student)
# print(student.name)
# print(student.age)

student_dict = student.model_dump() #model_dump() method to convert pydantic model to dictionary
print(type(student_dict))
print(student_dict)
print(student_dict["cgpa"])



student_json = student.model_dump_json() #model_dump_json() method to convert pydantic model to json
print(type(student_json))
print(student_json)



