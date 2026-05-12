from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def home():
    return {
        "message": "首页"
    }


@app.get("/user")
def get_user(name: str,age:int):
    return {
        "username": name,
        "age":age
    }

# @app.get("/student/{student_id}")
# def get_student(student_id: int):
#     return {
#         "当前的student_id是": student_id
#     }

@app.get("/student/{student_id}")
def get_student(student_id: int, name: str):
    return {
        "student_id": student_id,
        "name": name
    }


@app.get("/course/{course_id}")
def get_course(course_id:int,name:str):
    return{
        "获取的课程ID":course_id,
        "课程名字":name
    }