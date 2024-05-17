from transformers import BertTokenizer, BertModel
import torch

import os
import sys
import numpy as np
import javalang
from javalang.ast import Node
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
import pandas as pd



df= pd.read_csv('MLCQCodeSmellSamples.csv')  #Input csf file which contain link of the smelly and non-smelly java source code with their labels.


# -------------data frame cleaning like removing unnecesssary columns-----------------------
def df_cleaning(df_):
  print("data frame shape ",df_.shape)
  df_ = df_.dropna(subset=['link'])
  print("After removing NA - data frame shape ",df_.shape)

  df_.reset_index(inplace=True, drop=True)

  columns_to_keep = ['smell', 'severity', 'path','link']
  df_new= df_[columns_to_keep]

  return df_new

df_dir_path_name=df_cleaning(df)

# Load pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# ---------Extracting java code directly from the github using the link stored in the csv file------------
import requests
import re
def get_java_code(code_link):
  
    github_link = code_link
    # github_link = 'https://github.com/apache/syncope/blob/114c412afbfba24ffb4fbc804e5308a823a16a78/client/idrepo/ui/src/main/java/org/apache/syncope/client/ui/commons/ConnIdSpecialName.java/#L35-L37'

    # Convert GitHub link to raw content link
    raw_link = github_link.replace('/blob/', '/raw/')

    # Extract the starting and ending line numbers from the link
    line_range = github_link.split('#')[1]
    
    start_line, end_line = map(int, re.findall(r'\d+', line_range))
    print("start_line ",start_line)
    print("end_line ",end_line)
    # Send a GET request to fetch the content
    response = requests.get(raw_link)

    # Check if the request was successful
    if response.status_code == 200:
        # Split the content into lines
        lines = response.text.split('\n')

        # Extract lines within the specified range
        relevant_lines = lines[start_line-1:end_line]

        # Join the lines to get the code content
        
        code_content = " ".join(relevant_lines)
        
    else:
        print("Failed to fetch content from the GitHub link")
        # code_values.append(None)
        code_content=None
    return code_content

# Function to get embeddings
def get_bert_embeddings(sentences):
    # Tokenize the input sentences
    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")

    # Get the BERT model outputs
    with torch.no_grad():
        outputs = model(**inputs)

    # Get the embeddings (CLS token is used for sentence-level embeddings)
    embeddings = outputs.last_hidden_state[:, 0, :]
    embeddings1= torch.mean(embeddings,dim=0)
    # print("Initial Stage Embedding of all the paths extracted from the AST tree ",len(embeddings1))
    return embeddings1

def extract_all_paths(tree):
    stack = [(tree, [])]
    paths = []

    while stack:
        node, current_path = stack.pop()
        current_path.append((type(node).__name__, node))

        if hasattr(node, "children"):
            for child in reversed(node.children):
                stack.append((child, current_path[:]))

        if not hasattr(node, "children"):
            paths.append(current_path)
    # print("All extracted paths \n", paths)
    return paths

# Helper function to convert AST nodes to text sentences
def ast_node_to_sentence(node):
    # Customize this function to convert the AST node to a text sentence
    # For this example, we are just using the string representation of the node
    return str(node)

def create_java_ast(source_code):
    try:
      tokens = javalang.tokenizer.tokenize(source_code)
      parser = javalang.parser.Parser(tokens)
      return parser.parse_member_declaration()
    except javalang.parser.JavaSyntaxError as e:
      print(f"Syntax error in the Java code: {e}")
      return None
    
def read_java_files(java_code_, index):
    all_Embeddings = []
    countTree=0
    count=0

    # code_embedding = get_bert_embeddings(java_code)
    ast_tree= create_java_ast(java_code_)

    ast_embedding=[]
    if ast_tree is not None:
      # all_paths = []
      paths = extract_all_paths(ast_tree)
      # all_paths.extend([paths])
      # print("Paths ",paths)
      sentences = []
      for one_path_in_ast in paths:
        # print("single path from the AST tree ",one_path_in_ast)
        single_path_string=[]
        count_token=0
        for token, node_instance in one_path_in_ast:
        
          if node_instance is not None:
              sentence = f"{token}: {ast_node_to_sentence(node_instance)}"
              single_path_string.append(sentence)
              
              count_token+=1      
      sentences.append(single_path_string)
    
      path_embedding= get_bert_embeddings(sentences[0]) #[0]
    #   print("path_embedding ",len(path_embedding))
      # all_Embeddings.append(path_embedding)  # [0] used to get only 768 vector
      all_Embeddings= path_embedding
      countTree+=1
    else:
      count+=1
      countTree+=1
      print("AST tree not created for the index ",index)
      ast_embedding.append(None)
    return all_Embeddings


def getEmbedding(dataframe, path_column_name):
    """
    Read Java files from each directory path stored in the DataFrame.

    Parameters:
        dataframe (pd.DataFrame): The DataFrame containing the directory paths.
        path_column_name (str): The column name in the DataFrame containing the directory paths.

    Returns:
        None
    """
    count=0
    dir_embeddings=[]
    for index, row in dataframe.iterrows():
        print("------Extracting code No ", index," ---- Name of the codeSmell ", dataframe['smell'].loc[index]," -------")
        java_code = get_java_code(row[path_column_name])
        if java_code!=None:
          dir_embedding= read_java_files(java_code,index)
          print("dir_embedding ", len(dir_embedding))
          # print(dir_embedding)
          dir_embeddings.append(dir_embedding)
          print("\n")
        else:
          dir_embeddings.append([None])


        count+=1
        # if count==3:
        #     print("Only three dir")
        #     break
        # if index>=334:
          
        #   java_code = get_java_code(row[path_column_name])
        #   dir_embedding= read_java_files(java_code,index)
        #   print("dir_embedding ", len(dir_embedding))
        #   print(dir_embedding)
        #   dir_embeddings.append(dir_embedding)
        #   print("\n")


        #   count+=1
        #   if count==5:
        #     print("Only five dir")
        #     break
    print(" ",count, " code have been processed")
    return dir_embeddings

all_embedings= getEmbedding(df_dir_path_name, 'link')

# Convert tensor list into the list
all_embedings_list = []

for row in all_embedings:
    converted_row = []
    for tensor in row:
      if isinstance(tensor, torch.Tensor):
        tensor_list = tensor.tolist()
        converted_row.append(tensor_list)
      # else:
        # converted_row.append()
    all_embedings_list.append(converted_row)

df_dir_path_name['Embedding']= all_embedings_list

# df_dir_path_name1=df_dir_path_name.drop(df_dir_path_name.index[5:])
# df_dir_path_name1['Embedding']= all_embedings_list

for i in range(0,768):
    df_dir_path_name[str(i)]  = np.nan

for index,row in df_dir_path_name.iterrows():
    if index in []:
        continue
    # text_embed = model.encode_sentences([row["text"]], combine_strategy="mean")
    for i in range(768):
        if(len(df_dir_path_name['Embedding'][index])>0):
          df_dir_path_name.at[index, str(i)] = df_dir_path_name['Embedding'][index][i]



# # df_dir_path_name1.to_csv(output_file)
df_dir_path_name.to_csv("CSV_ASTBert_Output.csv")  #File which store the vector with respect to the each source file.