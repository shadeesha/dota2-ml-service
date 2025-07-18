import numpy as np
import os

from db.db_config import get_training_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

def multi_hot_encode(team_list, all_heroes):
    # team_list is a list or set of hero IDs (integers)
    return [1 if hero in team_list else 0 for hero in all_heroes]

def train():
    os.makedirs("model", exist_ok=True)

    df = get_training_data()

    print("Columns in training data:", df.columns.tolist())
    print(df.head())

    # List all heroes - update this if your hero IDs range is different
    ALL_HEROES = list(range(1, 139))

    # Convert string representation to list if necessary, or ensure list type
    # Example: if radiant_team is string like "{102,40,74,68}", parse it
    def parse_hero_list(s):
        if isinstance(s, str):
            return set(map(int, s.strip('{}').split(',')))
        return set(s)  # already list or set

    df['radiant_team'] = df['radiant_team'].apply(parse_hero_list)
    df['dire_team'] = df['dire_team'].apply(parse_hero_list)

    # Multi-hot encode both teams
    df['radiant_vec'] = df['radiant_team'].apply(lambda x: multi_hot_encode(x, ALL_HEROES))
    df['dire_vec'] = df['dire_team'].apply(lambda x: multi_hot_encode(x, ALL_HEROES))

    # Combine all features
    df['features'] = df.apply(
        lambda row: row['radiant_vec'] + row['dire_vec'] + [row['average_rank_tier']],
        axis=1
    )

    X = np.array(df['features'].tolist())
    y = df['target_hero'].astype(int)

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(x_train, y_train)

    joblib.dump(clf, "model/hero_model.pkl")

    y_pred = clf.predict(x_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))

if __name__ == "__main__":
    train()