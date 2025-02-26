# 通过 python -m evaluate.generate_question_context_pairs 运行此文件，否则有 ModuleNotFoundError

from llama_index.core import SimpleDirectoryReader
from llama_index.core.evaluation import generate_question_context_pairs
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai import OpenAI
from dotenv import load_dotenv
from config.rag_config import RAW_DATA_DIR, EVALUATE_DATA_DIR
import os


load_dotenv(".env")

llm = OpenAI(model="gpt-4o")

parser = SentenceSplitter()
documents = SimpleDirectoryReader(RAW_DATA_DIR).load_data()
nodes = parser.get_nodes_from_documents(documents)

qa_dataset = generate_question_context_pairs(nodes, llm=llm, num_questions_per_chunk=2)

# 需要提前创建 EVALUATE_DATA_DIR 目录
path = os.path.join(EVALUATE_DATA_DIR, "question_context_pairs.json")
# 生成的 json 中有乱码，但无需解决：https://github.com/run-llama/llama_index/issues/6415#issuecomment-1639114481
qa_dataset.save_json(path)
