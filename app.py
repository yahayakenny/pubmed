from Bio import Entrez
import xml.etree.ElementTree as ET
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import OpenAI

load_dotenv()

text_splitter = RecursiveCharacterTextSplitter(
    separators=[" ", ",", "\n"],
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False,
)


keyword_to_search = "Eflornithine in polycystic ovarian syndrome(PCOS)"


def search_and_fetch_from_pubmed(keyword, max_results=10):
    Entrez.email = "y.hkehinde@yahoo.com"
    search_handle = Entrez.esearch(db="pmc", term=keyword, retmax=max_results)
    search_results = Entrez.read(search_handle)
    search_handle.close()

    articles_info = []

    # Fetch full articles for each result
    for pmc_id in search_results["IdList"]:
        metadata_handle = Entrez.efetch(
            db="pmc", id=pmc_id, rettype="xml", retmode="xml"
        )
        metadata_data = metadata_handle.read()
        metadata_handle.close()

        full_article_handle = Entrez.efetch(db="pmc", id=pmc_id, retmode="xml")
        full_article_data = full_article_handle.read()
        full_article_handle.close()

        root = ET.fromstring(full_article_data)

        # Extract title from article XML
        title_element = root.find(".//article-title")
        article_title = title_element.text if title_element is not None else ""

        # Extract DOI from article XML
        doi_element = root.find(".//article-id[@pub-id-type='doi']")
        doi = doi_element.text if doi_element is not None else ""

        articles_info.append(
            {"title": article_title, "full_article": full_article_data, "doi": doi}
        )

    return articles_info


def extract_text_from_xml(xml_data):
    root = ET.fromstring(xml_data)
    text_content = " ".join(root.itertext())
    return text_content


def extract_article_info(article):
    article_text = extract_text_from_xml(article["full_article"])
    doi = article["doi"]

    return {
        "article_name": article["title"],
        "article_link": f"https://doi.org/{doi}",
        "article_text": article_text,
    }


def map_pubmed_artciles(keyword_to_search):
    result_articles = search_and_fetch_from_pubmed(keyword_to_search, max_results=10)
    pubmed_data = [extract_article_info(article) for article in result_articles]

    article_mapping = []
    chunks_arr = []

    for i in pubmed_data:
        article_chunk = text_splitter.split_text(i["article_text"])
        chunks_arr.append(article_chunk)

    print("chunks array", chunks_arr)

    return chunks_arr

    # for i in pubmed_data:
    #     article_name = i["article_name"]
    #     article_link = i["article_link"]
    #     article_chunk = text_splitter.split_text(i["article_text"])
    #     for chunk in article_chunk:
    #     article_mapping.append(
    #         {
    #             "article_chunk": i["article_text"],
    #             # "article_id": f"article_{str(uuid.uuid4())}",
    #             "article_name": article_name,
    #             "article_link": article_link,
    #         }
    #     )
    # return article_mapping


knowledge_base = map_pubmed_artciles(keyword_to_search)


def remove_outer_array(arr):
    if arr and isinstance(arr[0], list):
        return arr[0]
    else:
        return arr


inner_array = remove_outer_array(knowledge_base)

embeddings = OpenAIEmbeddings()
db = FAISS.from_texts(inner_array, embeddings)


if len(knowledge_base) > 0:
    docs = db.similarity_search(keyword_to_search)
    llm = OpenAI()
    chain = load_qa_chain(llm, chain_type="stuff")
    print(
        "final response>>>>>",
        chain.run(input_documents=docs, question={keyword_to_search}),
    )
