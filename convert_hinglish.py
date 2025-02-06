import os
import time
from groq import Groq
import json
from dotenv import load_dotenv
from loguru import logger
from tqdm import tqdm  # Progress bar
from extract_english import get_data_df, save_data, DATASET_LINK

# load env variables
load_dotenv()
MAX_RETRIES = 3

# Set sink for logger
logger.add("convert_hinglish.log", rotation="10 MB", backtrace=True, diagnose=True, filter=lambda record: "logFile" in record["extra"], level="INFO")

client = Groq(api_key=os.getenv("API_KEY"))

translation_count = 0

def convert_english_to_hinglish(text):
    global translation_count
    translation_count += 1

    # Convert English to Hinglish
    prompt = f"""Translate the following English and Chinese text into Hinglish (a mix of Hindi and English).
    Hinglish should use **only English characters**, but the words can be from both Hindi and English.
    The translation should **preserve the exact meaning** and should sound **natural and conversational** in Hinglish.

    English or Chinese: "{text}"
    Hinglish:"""
    # Call OpenAI GPT API
    retries = 0
    while retries < MAX_RETRIES: 
        try:
            time.sleep(3)  # Sleep for 3 seconds to avoid rate limit, no paisa
            response = client.chat.completions.create(
                model="deepseek-r1-distill-llama-70b", # may change to use OpenAI's GPT-3.5 model
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an expert translator who translates English or Chinese to Hinglish while keeping the meaning intact, using only English characters.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=1024,
                temperature=0.7,
                reasoning_format="hidden"
            )
            hinglish_text = response.choices[0].message.content.strip()
            # hinglish_text = "yeh ek test hai"
            # logger.info(f"input text: {text} | response: {hinglish_text}")
            if translation_count % 20 == 0:
                logger.bind(logFile=True).info(f"Translaton count: {translation_count} | input text: {text} | response: {hinglish_text}")
                

            return hinglish_text
        except Exception as e:
            if "rate limit" in str(e).lower():
                wait_time = (2 ** retries) * 1  # Exponential backoff
                logger.info(f"Rate limit reached. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                retries += 1
            else:
                logger.error(f"Error translating text: {text[:50]}... | Error: {e}")
                return None 


if __name__ == "__main__":
    # Load the data
    df = get_data_df("dataset_items_test.json")

    # taken random 1000 samples
    df = df.sample(1000, random_state=42)

    hinglish_captions = []
    for caption in tqdm(df['caption'], desc="Translating"):
        hinglish_captions.append(convert_english_to_hinglish(caption))

    # Convert English captions to Hinglish
    df["hinglish_caption"] = hinglish_captions

    # Save the data
    save_data(df, "dataset_items_test_hinglish.json")

    # Save to csv
    df.to_csv(os.path.join(DATASET_LINK, "dataset_items_test_hinglish.csv"), index=False)

    # text = "Unemployment has recently achieved the lowest rate in years."
    # convert_english_to_hinglish(text)
