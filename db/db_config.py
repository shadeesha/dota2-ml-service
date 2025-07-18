from sqlalchemy import create_engine
import pandas as pd

def get_training_data():
    engine = create_engine("postgresql+psycopg2://postgres:123@localhost:5432/dotadb")
    query = "SELECT * FROM training_data"
    df = pd.read_sql(query, engine)
    return df