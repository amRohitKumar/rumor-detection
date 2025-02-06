# Importing the required libraries
import os
import json
import pandas as pd
from langdetect import detect

DATASET_LINK = "e:\data.tar\data"
DATASET_NAME = "data"

def get_data_df(file_name):
    with open(os.path.join(DATASET_LINK, file_name), encoding='utf-8') as file:
        data = json.load(file)

    df = pd.DataFrame.from_dict(data, orient='index')
    return df

# Function to detect language

# Filter DataFrame for English captions
def filter_english_captions(df):

    def is_english(text):
        try:
            return detect(text) == 'en'
        except:
            return False 
    
    return df[df['caption'].apply(is_english)]

def save_data(df, file_name):
    data = df.to_dict(orient="index")
    with open(os.path.join(DATASET_LINK, file_name), "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


def extract_english_captions(file_name, save_name):
    # Load the data
    print("file_name: ", file_name)
    df = get_data_df(file_name)
    print("Initial length: ", len(df))

    # Filter English captions
    df_english = filter_english_captions(df)
    print("Final length: ", len(df_english))

    # Save the data
    save_data(df_english, save_name)

    return df_english

if __name__ == "__main__":
    extract_english_captions("dataset_items_test.json", "dataset_items_test_english.json")
    

