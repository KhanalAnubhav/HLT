# OCR imports
import os
import json
os.environ['USE_TORCH'] = '1'
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import math
# Langchain Imports
from langchain_community.vectorstores import FAISS
# from langchain_openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

from langchain_community.llms import CTransformers
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.llms.llamafile import Llamafile

# open ai api key
# os.environ['OPENAI_API_KEY'] = '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'



# get list of all the files
def list_files(directory):
    # Get the list of files in the specified directory
    files_list = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    return files_list

def convert_coordinates(geometry, page_dim):
        len_x = page_dim[1]
        len_y = page_dim[0]
        (x_min, y_min) = geometry[0]
        (x_max, y_max) = geometry[1]
        x_min = math.floor(x_min * len_x)
        x_max = math.ceil(x_max * len_x)
        y_min = math.floor(y_min * len_y)
        y_max = math.ceil(y_max * len_y)
        return [x_min, y_min, x_max, y_max]
def get_coordinates(output,file):
        page_dim = output['pages'][0]["dimensions"]
        text_coordinates = []
        with open(f"{file}.txt","+a") as f:
            for obj1 in output['pages'][0]["blocks"]:
                for obj2 in obj1["lines"]:
                    for obj3 in obj2["words"]:                
                        converted_coordinates = convert_coordinates(
                                                   obj3["geometry"],page_dim
                                                  )
                        converted_coordinates=','.join(str(v) for v in converted_coordinates)
                        f.write("{}: {}\n".format(converted_coordinates,
                                              obj3["value"]
                                              )
                             )
                        text_coordinates.append(converted_coordinates)
        return text_coordinates



# change this information
test_file_path = "/home/anubhav/datasets/invoice_1/train/images/2b8dde51-be3e-4e5e-8495-cd730a0089cc.png"
directory_path = "/home/anubhav/datasets/invoice_1/train/images"
ocr_path = "/home/anubhav/datasets/invoice_1/train/images/new_ocr"
# ocr_path = "/home/anubhav/datasets/invoice_1/train/ocr"
# json_path = "/home/anubhav/datasets/invoice_1/train/json"
# files_list = list_files(directory_path)

# Define a function to remove fields recursively
# def remove_fields(obj, fields):
#     if isinstance(obj, list):
#         for item in obj:
#             remove_fields(item, fields)
#     elif isinstance(obj, dict):
#         for key in list(obj.keys()):
#             if key in fields:
#                 del obj[key]
#             else:
#                 remove_fields(obj[key], fields)

def extract_md(directory_path, file_path):
    model = ocr_predictor(det_arch = 'db_resnet50',    
                        reco_arch = 'crnn_vgg16_bn', 
                        pretrained = True,
                        )
    img_path = f"{directory_path}/{file_path}"
    img = DocumentFile.from_images(img_path)
    result = model(img)

    # def extract_file_name(file_path):
    #     parts = file_path.split(os.path.sep)
    #     return parts[-1]

    output = result.export()
    # file_name = extract_file_name(file_path)
    # print(file_name)
    graphical_coordinates = get_coordinates(output,f'{directory_path}/new_ocr/{file_path}')
    print(type(graphical_coordinates))
    pass

# def extract_info(directory_path, file_path):
#     doc = DocumentFile.from_images(f'{directory_path}/{file_path}')
#     print(f"Number of pages: {len(doc)}")
#     predictor = ocr_predictor(det_arch = 'db_resnet50',    
#                         reco_arch = 'crnn_vgg16_bn', 
#                         pretrained = True,
#                         )
#     result = predictor(doc)
#     json_export = result.export()
#     fields_to_remove = ['confidence', 'page_idx', 'dimensions', 'orientation', 'language', 'artefacts']
#     # Remove the specified fields
#     remove_fields(json_export, fields_to_remove)

#     # Remove 'geometry' from 'blocks' and 'lines'
#     for page in json_export['pages']:
#         for block in page['blocks']:
#             if 'geometry' in block:
#                 del block['geometry']
#             for line in block.get('lines', []):
#                 if 'geometry' in line:
#                     del line['geometry']

#     # Convert the modified data back to JSON
#     modified_json = json.dumps(json_export, separators=(',', ':'))

#     # Print the modified JSON
#     print(modified_json)

#     #Convert the JSON data to a string
#     json_export_str = str(modified_json)

#     # Write the JSON data to a file
#     with open(f"{directory_path}/ocr/{file_path}.txt", "w") as file:
#         file.write(json_export_str)


def infer_llm(ocr_path, file):
    file_name = f'{ocr_path}/{file}.txt'
    # llm = CTransformers(
    #     model = "meta/Llama-2-7B-Chat-GGML",
    #     model_type="llama",
    #     max_new_tokens = 512,
    #     temperature = 0
    # )
    # Task: Analyze the JSON receipt data provided and group "value" entries with similar "geometry" proximity under "words," then summarize this information into one concise sentence.
    PROMPT_TEMPLATE = """

    Given the context respond to the question of the user.
    ```    
    Context:
    {context}
    ```
    ``` 
    Question: 
    {question}
    ```   
    Respond to the user in JSON format in key-value pairs.

    """

    QA_PROMPT = PromptTemplate(
        template=PROMPT_TEMPLATE, input_variables=['context', 'question']
    )
    # embedding_model = OpenAIEmbeddings(chunk_size=10)
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs = {'device':'cuda'}
    )
    OCR_content = TextLoader(file_name).load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap = 50)
    contents = text_splitter.split_documents(OCR_content)
    
    faiss_db = FAISS.from_documents(contents, embedding_model)
    retriever = faiss_db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    qa_chain = RetrievalQA.from_chain_type(
        llm = Llamafile(),
        retriever = retriever,
        chain_type_kwargs = {'prompt': QA_PROMPT},
        verbose = True
    )
    question = """

    Please extract the following details:
    Invoice number,
    Issue Data,
    Total
    """
    result = qa_chain({'query': question})
    print('\n ---- \n')
    print('filename', file)
    print(result['result'])
    # result_data = json.loads(result['result'])
    # with open(f'{json_path}/{file}.json', 'w') as f:
    #     json.dump(result_data, f, indent = 2)


# the process of extracting the information from the image file is completed so commenting this section of the code
# extract_md(directory_path, '2b8dde51-be3e-4e5e-8495-cd730a0089cc.png')
# /home/anubhav/datasets/invoice_1/train/images/d056cb61-069a-4d34-aeef-a9fd615ba6b6.jpg
# extract_md(directory_path, 'd056cb61-069a-4d34-aeef-a9fd615ba6b6.jpg')
# for the individual files in the new folder run the llm and store the result as json
# infer_llm(ocr_path, '2b8dde51-be3e-4e5e-8495-cd730a0089cc.png')

infer_llm(ocr_path, 'd056cb61-069a-4d34-aeef-a9fd615ba6b6.jpg')
