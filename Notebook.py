import os
import pandas as pd
from openai import OpenAI

# Initialize OpenAI client using environment variable for the API key.
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def load_and_merge_datasets(nasdaq_file: str, price_change_file: str) -> pd.DataFrame:
    """
    Load NASDAQ-100 companies and their YTD price change data, then merge them on the 'symbol' column.
    """
    nasdaq_df = pd.read_csv(nasdaq_file)
    price_df = pd.read_csv(price_change_file)
    # Merge on 'symbol' and keep only companies present in both datasets.
    merged_df = nasdaq_df.merge(price_df[["symbol", "ytd"]], on="symbol", how="inner")
    return merged_df

def classify_sector(symbol: str) -> str:
    """
    Use OpenAI's GPT-3.5 Turbo to classify the company symbol into one of the following sectors:
    Technology, Consumer Cyclical, Industrials, Utilities, Healthcare, Communication, Energy,
    Consumer Defensive, Real Estate, Financial.
    Returns only the sector name.
    """
    prompt = (
        f"Classify company {symbol} into one of the following sectors. "
        "Answer only with the sector name: Technology, Consumer Cyclical, Industrials, Utilities, "
        "Healthcare, Communication, Energy, Consumer Defensive, Real Estate, Financial."
    )
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    sector = response.choices[0].message.content.strip()
    return sector

def enrich_sector_information(df: pd.DataFrame) -> pd.DataFrame:
    """
    Loop through each company symbol in the DataFrame and classify its sector.
    The resulting sector is added as a new column.
    """
    # Apply the classification for each company symbol.
    df["Sector"] = df["symbol"].apply(classify_sector)
    return df

def get_stock_recommendations(df: pd.DataFrame) -> str:
    """
    Create a prompt containing the enriched NASDAQ-100 DataFrame and request
    a summary of stock performance YTD along with recommendations of the three best sectors
    and three or more companies per sector.
    Returns the model's response as a string.
    """
    prompt = (
        "Provide summary information about Nasdaq-100 stock performance year to date (YTD), "
        "recommending the three best sectors and three or more companies per sector. "
        "Company data: " + df.to_string(index=False)
    )
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    recommendations = response.choices[0].message.content.strip()
    return recommendations

def main():
    # Load and merge the datasets.
    nasdaq100_df = load_and_merge_datasets("nasdaq100.csv", "nasdaq100_price_change.csv")
    
    # Enrich the DataFrame by classifying each company's sector.
    enriched_df = enrich_sector_information(nasdaq100_df)
    
    # Display sector counts.
    print("Sector counts:")
    print(enriched_df["Sector"].value_counts())
    
    # Get and print stock recommendations.
    recommendations = get_stock_recommendations(enriched_df)
    print("\nStock Recommendations:")
    print(recommendations)

if __name__ == "__main__":
    main()
