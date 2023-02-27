from sqlalchemy import Column, TEXT, INT, BIGINT
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class Test(Base):
    __tablename__ = "test"
    id = Column(BIGINT, nullable = False, autoincrement = True, primary_key = True)
    name = Column(TEXT, nullable=False)
    number = Column(INT, nullable=False)
    