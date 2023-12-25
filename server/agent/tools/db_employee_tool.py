from langchain.pydantic_v1 import BaseModel, Field


class PersonSchema(BaseModel):
    """获得指定人员的信息"""

    # user_id: str = Field(description="用户编号，例如：1234567，6543213")
    name: str = Field(description="姓名，例如：zhangsan，admin，张三，王五，李四")


# 方法内部可以替换成远程访问
def query_employee(name: str) -> str:
    """获得指定人员的信息"""

    print("\nname " + name)

    return "这是一个人的信息 " + name
