import re
import json
import nltk
import argparse
import hashlib
import flask
import random
from collections import OrderedDict
import math
nltk.download('punkt')
nltk.download('stopwords')
from indexer import Indexer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from flask import Flask, request, jsonify
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
import time
import math

class Node:
    def __init__(self, data):
        self.data = data
        self.next = None
        self.skip = None   

class LinkedList:
    def __init__(self):
        self.head = None

    def __len__(self):
        current = self.head
        count = 0
        while current:
            count += 1
            current = current.next
        return count

    def append(self, data):
        new_node = Node(data)
        if not self.head:
            self.head = new_node
            return
        
        last_node = self.head
        prev_skip_node = None
        count = 0
        skip_interval = math.isqrt(len(self))
        
        while last_node.next:
            count += 1
            if count == skip_interval:
                if prev_skip_node:
                    prev_skip_node.skip = last_node
                prev_skip_node = last_node
                count = 0
            last_node = last_node.next
        
        last_node.next = new_node

    def contains(self, data):
        current_node = self.head
        while current_node:
            if current_node.data == data:
                return True
            if current_node.skip and current_node.skip.data <= data:
                current_node = current_node.skip
            else:
                current_node = current_node.next
        return False

    def to_list(self):
        list_data = []
        current_node = self.head
        while current_node:
            list_data.append(current_node.data)
            current_node = current_node.next
        return list_data


'''

def preprocess(text):
    # Convert to lowercase
    text = text.lower()
    
    # Replace hyphens, en dashes, and other similar characters with double spaces
    text = text.replace("-", "  ").replace("–", "  ").replace("‐", "  ").replace("/", "  ").replace("–", "  ")

    # Remove other special characters, retaining only alphabets, numbers, and whitespaces
    text = re.sub(r'[^a-z0-9\s]', '', text)


    # Merge consecutive spaces into a single space
    text = re.sub(r'\s+', ' ', text).strip()

    print(text)

    # Tokenize on white spaces
    tokens = re.split(r'\s+', text)

    # Remove stop words and stem (assuming stemmer and stop_words are defined elsewhere in your code)
    tokens = [stemmer.stem(token) for token in tokens if token not in stop_words]

    return tokens

'''

def preprocess(text):
    # Convert to lowercase
    text = text.lower()
    
    # Replace hyphens, en dashes, em dashes, and other similar characters with single spaces
    text = text.replace("-", " ").replace("–", " ").replace("—", " ").replace("‐", " ").replace("/", " ")

    # Remove other special characters, retaining only alphabets, numbers, and whitespaces
    text = re.sub(r'[^a-z0-9\s]', '', text)

    # Merge consecutive spaces into a single space
    text = re.sub(r'\s+', ' ', text).strip()

    #print(text)

    # Tokenize on white spaces
    tokens = re.split(r'\s+', text)

    # Remove stop words and stem
    tokens = [stemmer.stem(token) for token in tokens if token not in stop_words]

    return tokens




def get_item_from_linked_list(ll, position):
    current_node = ll.head  
    index = 0
    while current_node:
        if index == position:
            return current_node.data  
        current_node = current_node.next
        index += 1
    return None

def create_index(corpus):
    index = OrderedDict()
    for doc_id, doc in enumerate(corpus):
       
        
        for term in preprocess(doc):
            if term in index:
                if not index[term].contains(doc_id):
                    index[term].append(doc_id)
            else:
                index[term] = LinkedList()
                index[term].append(doc_id)
    return index


def linked_list_length(ll):
    if isinstance(ll, list):
        return len(ll)
    
    length = 0
    current_node = ll.head
    while current_node:
        length += 1
        current_node = current_node.next
    return length
def get_nth_node(ll, n):
    current = ll.head
    for _ in range(n):
        if not current:
            return None
        current = current.next
    return current

def read_corpus_from_file(filename):
    with open(filename, 'r') as file:
        return [line.strip() for line in file.readlines()]

def daat_and_merge(*posting_lists):
    posting_lists = sorted(posting_lists, key=linked_list_length)  
    
    pointers = [0] * len(posting_lists)
    results = []
    comparisons = 0

    while True:
       
        current_docs = [get_item_from_linked_list(lst, pointers[i]) if pointers[i] < linked_list_length(lst) else None for i, lst in enumerate(posting_lists)]

      
        if None in current_docs:
            break

       
        if len(set(current_docs)) == 1:
            results.append(current_docs[0])
            pointers = [x + 1 for x in pointers]
            comparisons += len(posting_lists) - 1
        else:
            max_doc = max(current_docs)
            for i, doc in enumerate(current_docs):
                while pointers[i] < linked_list_length(posting_lists[i]) and get_item_from_linked_list(posting_lists[i], pointers[i]) < max_doc:
                    pointers[i] += 1
                    comparisons += 1

    return results, comparisons

def get_with_skip(ll):
    """
    Retrieve the postings list using skip pointers.
    This function assumes that the LinkedList nodes have a `skip` attribute that can be None or point to another node.
    """
    if not isinstance(ll, LinkedList):
        raise TypeError(f"Expected 'LinkedList', got '{type(ll).__name__}'")

    current_node = ll.head
    postings_with_skip = []

    while current_node:
        postings_with_skip.append(current_node.data)
        if hasattr(current_node, 'skip') and current_node.skip:
            current_node = current_node.skip
        else:
            current_node = current_node.next  

    return postings_with_skip


def get_postings_with_skip(terms):
    """
    For each term, get the postings list using skip pointers.
    """
    postings_dict = {}

    for term in terms:
        preprocessed_term = preprocess(term)[0]  
        if preprocessed_term in index:  
            postings_with_skip = get_with_skip(index[preprocessed_term])
            if postings_with_skip:
                postings_dict[preprocessed_term] = postings_with_skip
            else:
                postings_dict[preprocessed_term] = []

    return postings_dict


app = Flask(__name__)


corpus_filename = 'input_corpus.txt'
corpus = read_corpus_from_file(corpus_filename)
index = create_index(corpus)
total_docs = len(corpus)
daat_results_with_tf = {}
sample_text = "This is a SARS-novel test."
def daat_and_merge_with_skip(terms, *posting_lists):
    pointers = [0 for _ in posting_lists]
    result_docs = []
    comparisons = 0
    
    
    skip_lengths = [int(len(linked_list_to_list(pl))**0.5) for pl in posting_lists]

    while all(pointers[i] < len(linked_list_to_list(posting_lists[i])) for i in range(len(posting_lists))):
        
        doc_ids = [get_nth_node(posting_lists[i], pointers[i]).data for i in range(len(posting_lists))]
        if not doc_ids:
            break
        min_doc_id = min(doc_ids)
        comparisons += len(doc_ids) - 1

        if all(doc_id == min_doc_id for doc_id in doc_ids):
            result_docs.append(min_doc_id)
            for i in range(len(posting_lists)):
                current_node = get_nth_node(posting_lists[i], pointers[i])
                
                while hasattr(current_node, "skip") and current_node.skip and current_node.skip.data <= min_doc_id:

                    current_node = current_node.skip
                    pointers[i] += skip_lengths[i]
                pointers[i] += 1
        else:
            for i in range(len(posting_lists)):
                current_node = get_nth_node(posting_lists[i], pointers[i])
                while hasattr(current_node, "skip") and current_node.skip and current_node.skip.data <= min_doc_id:

                    current_node = current_node.skip
                    pointers[i] += skip_lengths[i]
                if current_node.data == min_doc_id:
                    pointers[i] += 1

    return result_docs, comparisons

def daat_and_merge_with_skipSORTED(terms, total_docs, doc_term_frequencies, *posting_lists):
    pointers = [0 for _ in posting_lists]
    result_docs = []
    comparisons = 0
    skip_lengths = [int(len(linked_list_to_list(pl))**0.5) for pl in posting_lists]

    while all(pointers[i] < len(linked_list_to_list(posting_lists[i])) for i in range(len(posting_lists))):
        doc_ids = [get_nth_node(posting_lists[i], pointers[i]).data for i in range(len(posting_lists))]
        
        min_doc_id = min(doc_ids)
        comparisons += len(doc_ids) - 1

        if all(doc_id == min_doc_id for doc_id in doc_ids):
            result_docs.append(min_doc_id)
            for i in range(len(posting_lists)):
                current_node = get_nth_node(posting_lists[i], pointers[i])
                
                while hasattr(current_node, "skip") and current_node.skip and current_node.skip.data <= min_doc_id:

                    current_node = current_node.skip
                    pointers[i] += skip_lengths[i]
                pointers[i] += 1

        else:
            for i in range(len(posting_lists)):
                current_node = get_nth_node(posting_lists[i], pointers[i])
                
                while hasattr(current_node, "skip") and current_node.skip and current_node.skip.data <= min_doc_id:

                    current_node = current_node.skip
                    pointers[i] += skip_lengths[i]
                if current_node.data == min_doc_id:
                    pointers[i] += 1

    tf_idf_scores = {}
    for doc_id in result_docs:
        score = 0
        for i, term in enumerate(terms):
            tf = doc_term_frequencies.get(doc_id, {}).get(term, 0)
            df = len(linked_list_to_list(posting_lists[i]))
            idf = math.log(total_docs / df) if df > 0 else 0
            score += tf * idf
        tf_idf_scores[doc_id] = score

    sorted_results = sorted(tf_idf_scores.keys(), key=lambda x: tf_idf_scores[x], reverse=True)
    
    return sorted_results, comparisons

def tf_idf(term, doc_id, total_docs):
    tf = 1  
    df = linked_list_length(index[term])
    idf = math.log(total_docs / df)
    return tf * idf

def list_to_linked_list(lst):
    linked_list = LinkedList()
    for item in lst:
        linked_list.append(item)
    return linked_list

def linked_list_to_list(ll, use_skip=False):
    """Convert a LinkedList to a Python list. If use_skip is True, use skip pointers where available."""
    if isinstance(ll, list):
        ll = list_to_linked_list(ll)
    current_node = ll.head
    result = []
    while current_node:
        result.append(current_node.data)
        if use_skip and hasattr(current_node, "skip") and current_node.skip:
            current_node = current_node.skip
        else:
            current_node = current_node.next
    return result


def smallest_doc_id(converted_posting_lists, pointers):
    return min(converted_posting_lists[i][pointers[i]] for i in range(len(converted_posting_lists)) if pointers[i] < len(converted_posting_lists[i]))


def daat_and_merge_tf_idf(terms, total_docs, corpus, posting_lists):
    pointers = [0 for _ in posting_lists]
    result_docs = []
    comparisons = 0
    doc_idds = list(corpus.keys())
    converted_posting_lists = [linked_list_to_list(pl) for pl in posting_lists]

    while not all(pointer >= len(pl) for pointer, pl in zip(pointers, converted_posting_lists)):
        current_docs = [pl[pointer] if pointer < len(pl) else float('inf') for pointer, pl in zip(pointers, converted_posting_lists)]
        min_doc_id = min(current_docs)
        min_doc_id_indices = [i for i, doc_id in enumerate(current_docs) if doc_id == min_doc_id]

        comparisons += len(pointers) - 1  # for finding min doc ID

        if all(doc_id == min_doc_id for doc_id in current_docs):
            doc_name = doc_idds[min_doc_id]
            score = sum(tf_idf(term, doc_name, total_docs) for term in terms)
            result_docs.append((doc_name, score))
            pointers = [p + 1 for p in pointers]  # move all pointers
        else:
            for i in min_doc_id_indices:
                pointers[i] += 1  # move only the pointers that were at the min doc ID

    result_docs.sort(key=lambda x: x[1], reverse=True)
    sorted_doc_ids = [int(doc_id) for doc_id, _ in result_docs]
    return sorted_doc_ids, comparisons




def read_corpus_from_file(filename):
    with open(filename, 'r') as f:
        return {line.split()[0]: ' '.join(line.split()[1:]) for line in f.readlines()}

corpus = read_corpus_from_file(corpus_filename)


def generate_doc_term_frequencies(corpus):
    doc_term_freq = {}
    for doc_id, doc_content in corpus.items():
        terms_in_doc = preprocess(doc_content) 
        term_freq = {}
        for term in terms_in_doc:
            term_freq[term] = term_freq.get(term, 0) + 1
        doc_term_freq[doc_id] = term_freq
    return doc_term_freq

doc_term_frequencies = generate_doc_term_frequencies(corpus)

def sanity_checker(command):
    """ DO NOT MODIFY THIS. THIS IS USED BY THE GRADER. """

    index = create_index(corpus)
    kw = random.choice(list(index.keys()))

    indexer_instance = Indexer()  # instantiate the Indexer class to get its type

    return {
        "index_type": str(type(index)),
        "indexer_type": str(type(indexer_instance)),
        "post_mem": str(index[kw]),
        "post_type": str(type(index[kw])),
        "node_mem": str(index[kw].head),
        "node_type": str(type(index[kw].head)),
        "node_value": str(index[kw].head.data) if index[kw].head else "None",
        "command_result": eval(command) if "." in command else ""
    }

def get_postings_with_skip(postingsss, skip_interval):
    postings_with_skip = []
    index = 0
    while index < len(postingsss):
        # Add the current index
        postings_with_skip.append(postingsss[index])
        # Skip the desired interval and move to the next
        index += skip_interval + 1
    return postings_with_skip


@app.route("/execute_query", methods=['POST'])
def execute_query():
    start_time = time.time()

    request_data = request.get_json()
    if not request_data or 'queries' not in request_data:
        return jsonify({"error": "Missing 'queries' in request data"}), 400

    queries = request_data['queries']

    postings = {}
    daat_results = {}
    postings_with_skip = {}  
    doc_term_frequencies = {}
    daat_results_with_skip = {}
    daat_results_skipSORTED = {}
    dand = {}
    corpusss = read_corpus_from_file(corpus_filename)
    doc_idds = list(corpusss.keys())

    for query in request_data['queries']:
        terms = preprocess(query)
        #print(f"Query Terms for '{query}': {terms}")
        posting_lists = [index.get(term, []) for term in terms]
        matched_dand, comparisons_dand = daat_and_merge(*posting_lists)
        matched_dand = [int(doc_idds[i]) for i in matched_dand]
        matched_doc_ids_with_skip, comparisons_with_skip = daat_and_merge_with_skip(terms, *posting_lists)
        matched_doc_ids_with_skip = [int(doc_idds[i]) for i in matched_doc_ids_with_skip]
        matched_doc_ids_tf_idf, comparisons_tf_idf = daat_and_merge_tf_idf(terms, total_docs,corpusss, posting_lists)
        matched_doc_ids_with_skipSORTED, comparisons_with_skipSORTED = daat_and_merge_with_skipSORTED(terms, total_docs,doc_term_frequencies, *posting_lists)
        matched_doc_ids_with_skipSORTED = [int(doc_idds[i]) for i in matched_doc_ids_with_skipSORTED]
        dand[query] = {
            'results': matched_dand,
            'num_docs': len(matched_dand),
            'num_comparisons': comparisons_dand
        }
        
        daat_results_with_skip[query] = {
            'results': matched_doc_ids_with_skip,
            'num_docs': len(matched_doc_ids_with_skip),
            'num_comparisons': comparisons_with_skip
        }

        daat_results_with_tf[query] = {
            'results': matched_doc_ids_tf_idf,
            'num_docs': len(matched_doc_ids_tf_idf),
            'num_comparisons': comparisons_tf_idf
        }

        daat_results_skipSORTED[query] = {
            'results': matched_doc_ids_with_skipSORTED,
            'num_docs': len(matched_doc_ids_with_skipSORTED),
            'num_comparisons': comparisons_with_skipSORTED
        }
        corpusss = read_corpus_from_file(corpus_filename)

        doc_idds = list(corpusss.keys())
        for term in terms:
            if term in index:
                #print("termmmmmmmmmmm",term)
                #print(f"Term '{term}' found in index.")
               
                postings_list_indices = linked_list_to_list(index[term])
                postings_list = [int(doc_idds[i]) for i in postings_list_indices]
                postings_list = sorted(postings_list, key=int)
                postings[term] = postings_list
                '''
                sk = int(len(index[term])**0.5)

                print("first term",term)
                print("first sk:",sk)
                if sk**2 == len(index[term]):
                    print("Perfect term",term)
                    
                    sk -= 1
                print("term",term)
                print("sk:",sk)
                '''
                print(len(index[term]))
                sk = int(len(index[term])**0.5)
                print("first term", term)
                print("first sk:", sk)

# Check if the square of sk equals the length of index[term]
                if (sk + 1)**2 <= len(index[term]):
                    sk += 1
                else:
                    sk -= 1
                print("term", term)
                print("sk:", sk)  
                postingssss = list_to_linked_list(postings[term])
                posts = linked_list_to_list(postingssss)
                postings_list_skip = get_postings_with_skip(posts, sk)
                postings_with_skip[term] = postings_list_skip
                pp=postings_with_skip
            else:
                #print(f"Term '{term}' NOT found in index.")
                postings[term] = []
                postings_with_skip[term] = []

    linked_postings = {}
    for key, value in postings.items():
        ll = LinkedList()
        for item in value:
            ll.append(item)
        linked_postings[key] = ll

    linked_postings_with_skip = {}
    for key, value in postings_with_skip.items():
        ll = LinkedList()
        for item in value:
            ll.append(item)
        linked_postings_with_skip[key] = ll

    random_command = request_data.get("random_command", "")

    output_data = {
        "Response": {
            'daatAnd': dand,
            'daatAndSkip': daat_results_with_skip,
            'daatAndTfIdf': daat_results_with_tf,
            'daatAndSkipTfIdf': daat_results_skipSORTED,
            'postingsList': postings,
            'postingsListSkip': postings_with_skip,
            'sanity': sanity_checker(random_command),
            'time_taken': str(time.time() - start_time)
        }
    }

   
    with open(output_location, 'w') as outfile:
        json.dump(output_data, outfile)

    
    response = {
        "Response": output_data["Response"],
        "time_taken": str(time.time() - start_time),
        "username_hash": username_hash
    }
    

    return flask.jsonify(response)

if __name__ == "__main__":
    output_location = "project2_output.json"
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--corpus", type=str, help="Corpus File name, with path.")
    parser.add_argument("--output_location", type=str, help="Output file name.", default=output_location)
    parser.add_argument("--username", type=str,
                        help="Your UB username. It's the part of your UB email id before the @buffalo.edu. "
                             "DO NOT pass incorrect value here")

    argv = parser.parse_args()

    corpus = argv.corpus
    output_location = argv.output_location
    username_hash = hashlib.md5(argv.username.encode()).hexdigest()

    app.run(host="0.0.0.0", port=9999)
