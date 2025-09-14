from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_google_genai import ChatGoogleGenerativeAI
import os

import pandas as pd

df = pd.read_csv(
    "cleaned_SIH_report_from_excel.csv"
)

import os
os.environ["GOOGLE_API_KEY"] = "AIzaSyC8KURHfyCsXiUc8NszqgGu0b-ZFxV5s9U"

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)
agent = create_pandas_dataframe_agent(llm=llm, df=df, verbose=True, allow_dangerous_code=True)

print(agent.invoke("how many columns and rows are there?"))

