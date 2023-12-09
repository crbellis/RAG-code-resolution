import ast
import numpy as np
from typing import List
import requests
from sentence_transformers import SentenceTransformer
from sentence_transformers import util
import torch
from dotenv import load_dotenv
import pinecone
import os

load_dotenv()

pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment="gcp-starter")
index = pinecone.Index("poc")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
model = SentenceTransformer("all-MiniLM-L6-v2")


def parse_code(res):
    return res["response"].split("```")[1]  # buggy


def add_comment(block):
    prompt = f"Add comments to this code, wrap the output in a code block:\n```\n{block}\n```\n\n"
    res = call_llm(prompt)
    if res == None:
        return
    return parse_code(res)


def recommend_fix(block, issue):
    prompt = f"""There is an issue with this code: {issue}
    Fix this code, wrap the output in a code block:\n```\n{block}\n```\n\n"""
    res = call_llm(prompt)
    if res == None:
        return
    return parse_code(res)


def call_llm(prompt):
    res = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama2",
            "prompt": prompt,
            "stream": False,
        },
    )
    return res.json()


def embed_code_block(block):
    return model.encode(block)


def extract_code_blocks(file_path):
    with open(file_path, "r") as file:
        source_code = file.read()

    # Parse the source code into an Abstract Syntax Tree (AST)
    tree = ast.parse(source_code)

    code_blocks = []

    # Visit each node in the AST
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) or isinstance(node, ast.ClassDef):
            # Extract function or class code
            code_blocks.append(ast.get_source_segment(source_code, node))

    return code_blocks


def update_v_db(source_file, code_blocks):
    # might need to keep a reference of code blocks in db
    for i, block in enumerate(code_blocks, start=1):
        index.upsert([(f"{source_file}/block_{i}", block.tolist())])


def get_similar_blocks(issue_embedding: List[float]):
    # or locally
    """
    # Find the most similar code block
    similarities = util.pytorch_cos_sim(
        np.array(issue_embedding), np.array(encoded_blocks)
    )[0]
    """

    # lookup blocks from pinecone
    results = index.query(issue_embedding, top_k=10)
    if results["matches"]:
        return results["matches"][0]["id"]
    return


# Example usage
file_path = "dummy.py"
blocks = extract_code_blocks(file_path)

# Print the code blocks
for i, block in enumerate(blocks, start=1):
    print(f"Code Block {i}:\n{block}\n{'='*30}")


def extract_code_idx(name):
    return int(name.split("_")[1]) - 1


def main():
    commented_code_blocks = [add_comment(block) for block in blocks]
    # encoded_blocks = [embed_code_block(block) for block in commented_code_blocks]
    # update_v_db("dummy.py", encoded_blocks)

    issue = """I'm trying to run a calculation on a list of numbers but I'm getting some strange values.
    The list is [1, 2, 3]. I want to calculate the cube of each number.
    The expected output is [1, 8, 27] but I'm getting [1, 4, 9].
    """

    # Encode the issue
    issue_embedding = model.encode(issue)
    print(issue_embedding.shape)

    # lookup blocks from pinecone
    result = get_similar_blocks(issue_embedding.tolist())
    print(result)

    # Print the most similar code block
    # most_similar_block = commented_code_blocks[
    #     int(result.argmax())
    # ]  # get argmax of blocks
    # print(most_similar_block)

    idx = extract_code_idx(result)
    fix = recommend_fix(commented_code_blocks[idx], issue)
    print(fix)


if __name__ == "__main__":
    main()
