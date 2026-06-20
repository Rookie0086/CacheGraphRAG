import os
import sys
import json
import gzip
import argparse
import logging
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv
from data.paths import ECTQA_DATAPATH
from src.utils import file_exist

def load_documents_from_corpus(corpus_path: Path) -> List[Dict]:
    """
    Load documents from the ECT-QA corpus.
    
    Args:
        corpus_path: Path to the corpus file (base.jsonl.gz)
        
    Returns:
        List of document dictionaries
    """
    documents = []
    try:
        with gzip.open(corpus_path, 'rt', encoding='utf-8') as f:
            for i, line in enumerate(f):
                doc = json.loads(line)
                documents.append(doc)
        print(f"✅ Loaded {len(documents)} documents from corpus")
        return documents
    except Exception as e:
        raise RuntimeError(f"Error loading corpus: {e}")

def prepare_documents_for_insertion(documents: List[Dict]) -> List[str]:
    """
    Args:
        documents: List of documents (either from corpus or txt files)
        
    Returns:
        List of documents in format {"title": str, "doc": str}
    """
    if not documents:
        return []
    
    # Auto-detect format: check if first document has 'title' and 'doc' keys (text format)
    # or 'cleaned_content'/'raw_content' keys (corpus format)
    first_doc = documents[0]
    is_corpus_format = 'cleaned_content' in first_doc or 'raw_content' in first_doc
    
    if not is_corpus_format:
        # Already in the correct format (from txt files)
        # Just validate and return
        for doc in documents:
            if 'title' not in doc or 'doc' not in doc:
                raise ValueError(f"Document missing required keys 'title' or 'doc': {list(doc.keys())}")
        return documents
    
    # Process corpus format documents
    prepared_docs = []
    for doc in documents:
        content = doc.get('cleaned_content', doc.get('raw_content', ''))
        if not content:
            print(f"⚠️  Warning: Document {doc.get('company_name', 'Unknown')} has no content, skipping")
            continue
        
        # Create a descriptive title
        company = doc.get('company_name', 'Unknown')
        year = doc.get('year', '')
        quarter = doc.get('quarter', '')
        if year and quarter:
            title = f"{company} {year} Q{str(quarter)[-1]}"
        elif year:
            title = f"{company} {year}"
        else:
            title = company
        
        prepared_docs.append(f"title: {title}\ndoc: {content}")
    
    return prepared_docs

def get_ectqa_info(corpus_file: str = "base.jsonl.gz") -> List[Dict]:
    """
    Get ECT-QA information from the corpus file.
    
    Args:
        corpus_file: Name of the corpus file (default: base.jsonl.gz)

    Returns:
        List of document dictionaries
    """
    data_file = os.path.join(ECTQA_DATAPATH, f"corpus/{corpus_file}")
    assert file_exist(data_file), f"{data_file} not exist!"
    
    texts = []
    questions = []
    answers = []
    documents = load_documents_from_corpus(data_file)
    texts = prepare_documents_for_insertion(documents)
    qa_file = os.path.join(ECTQA_DATAPATH, "questions/local_new.jsonl")
    assert file_exist(qa_file), f"{qa_file} not exist!"
    with open(qa_file, "r") as f:
        for line in f:
            instance = json.loads(line)
            if instance.get("answer") == "unanswerable" or instance.get("evidence_list") == []:
                continue
            questions.append(instance["question"])
            answers.append(instance["answer"])
    data_info = {
        "texts": texts,
        "questions": questions,
        "answers": answers,
    }
    return data_info

if __name__ == "__main__":
    pass