from sqlalchemy import *
from sqlalchemy.orm import sessionmaker
from os import environ
from dotenv.main import load_dotenv

# .env 환경파일 로드
load_dotenv()

# 디비 접속 URL
DB_CONN_URL = "{}://{}:{}@{}:{}/{}".format(
    environ["DB_TYPE"],
    environ["DB_USER"],
    environ["DB_PASSWD"],
    environ["DB_HOST"],
    environ["DB_PORT"],
    environ["DB_NAME"],
)


class engineconn:
    def __init__(self):
        self.engine = create_engine(DB_CONN_URL, pool_recycle=500)

    def sessionmaker(self):
        Session = sessionmaker(bind=self.engine)
        session = Session()
        return session

    def connection(self):
        conn = self.engine.connect()
        return conn
