# Start your code here!
import os
import pandas as pd
from openai import OpenAI

# Instantiate an API client
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# Read in the two datasets
nasdaq100 = pd.read_csv("nasdaq100.csv")
price_change = pd.read_csv("nasdaq100_price_change.csv")

# Add symbol into nasdaq100
nasdaq100 = nasdaq100.merge(price_change[["symbol", "ytd"]], on="symbol", how="inner")

# Preview the combined dataset
nasdaq100.head()

# Loop through the NASDAQ companies
for company in nasdaq100["symbol"]:
    # Create a prompt to enrich nasdaq100 using OpenAI
    prompt = f'''Classify company {company} into one of the following sectors. Answer only with the sector name: Technology, Consumer Cyclical, Industrials, Utilities, Healthcare, Communication, Energy, Consumer Defensive, Real Estate, Financial.
'''
    # Create a request to the completions endpoint
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{ "role": "user", "content": prompt}],
        temperature=0.0,
    )
    # Store the output as a variable called sector
    sector = response.choices[0].message.content
    
    # Add the sector for the corresponding company
    nasdaq100.loc[nasdaq100["symbol"] == company, "Sector"] = sector
    
# Count the number of sectors
nasdaq100["Sector"].value_counts()

# Prompt to get stock recommendations
prompt = f'''Provide summary information about Nasdaq-100 stock performance year to date (YTD), recommending the three best sectors               and three or more companies per sector.
            Company data: {nasdaq100} 
'''

# Get the model response
response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{ "role": "user", "content": prompt}],
        temperature=0.0,
    )

# Store the output as a variable and print the recommendations
stock_recommendations = response.choices[0].message.content
print(stock_recommendations)
