from pydantic import EmailStr, BaseModel, Field
from typing import Optional

class person(BaseModel):
    
    Name : str 
    Email : EmailStr
    age : Optional[int] = None
    cgpa :  float = Field(gt = 0, lt = 7, description = 'Decimal Value representing the cgpa of the student')


new_student = {'Name' : 'Varad G', 'Email' : 'abc@gmail.com', 'age' : 24, 'cgpa' : 6.7 }

student = person(**new_student)

student_dict = dict(student)
student_json = student.model_dump_json()
print(student_dict)
    