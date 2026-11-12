from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()
# LLM
model=ChatOpenAI(model="gpt-5")
res=model.invoke("how r u?")
print(res.content)