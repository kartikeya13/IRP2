'''
import re
import math
from collections import defaultdict
from nltk.stem import PorterStemmer


# Custom stop words list (simple for the sake of the example)
stop_words = set(["i" ,"me", "my", "myself", "we", "our" ,"ours" ,"ourselves" ,"you" ,"your", "yours" ,"yourself" ,"yourselves", "he" ,"him" ,"his" ,"himself","she", "her", "hers" ,"herself" ,"it" ,"its" ,"itself" ,"they", "them" ,"their" ,"theirs" ,"themselves", "what", "which" ,"who" ,"whom", "this", "that", "these", "those", "am" ,"is", "are" ,"was", "were", "be" ,"been" ,"being" ,"have", "has" ,"had" ,"having", "do" ,"does", "did" ,"doing" ,"a" ,"an" ,"the" ,"and", "but" ,"if" ,"or", "because" ,"as" ,"until" ,"while", "of" ,"at" ,"by", "for" ,"with", "about", "against" ,"between", "into", "through" ,"during", "before", "after" ,"above", "below" ,"to" ,"from" ,"up" ,"down" ,"in" ,"out", "on" ,"off", "over", "under", "again", "further", "then", "once" ,"here", "there", "when", "where", "why" ,"how" ,"all", "any", "both" ,"each" ,"few" ,"more" ,"most", "other" ,"some", "such", "no" ,"nor" ,"not" ,"only", "own" ,"same", "so" ,"than" ,"too", "very", "s" ,"t" ,"can" ,"will", "just", "don" ,"should", "now"])

class ListNode:
    def __init__(self, doc_id, tf_idf):
        self.doc_id = doc_id
        self.tf_idf = tf_idf
        self.next = None
        self.skip = None

def process_document(document):
    # Convert to lowercase
    document = document.lower()
    # Remove special characters
    document = re.sub(r'[^a-z0-9 ]', '', document)
    # Remove extra spaces
    document = re.sub(r'\s+', ' ', document).strip()
    # Tokenize and remove stopwords
    tokens = [token for token in document.split() if token not in stop_words]
    # Apply stemming
    tokens = [stemmer.stem(token) for token in tokens]
    return tokens

# Load corpus
with open("input_corpus.txt", 'r') as f:
    corpus = [line.strip().split('\t') for line in f.readlines()]

# Initialize stemmer
stemmer = PorterStemmer()

# Build inverted index
inverted_index = defaultdict(list)


for doc_id, document in corpus:
    tokens = process_document(document)
    doc_length = len(tokens)
    token_freqs = defaultdict(int)
    for token in tokens:
        token_freqs[token] += 1
    for token, freq in token_freqs.items():
        tf = freq / doc_length
        node = ListNode(doc_id, tf)
        # Ensure postings are stored in a linked list and ordered by doc_id
        if inverted_index[token]:
            last_node = inverted_index[token][-1]
            last_node.next = node
            inverted_index[token].append(node)
        else:
            inverted_index[token].append(node)


# Calculate IDF and TF-IDF
num_docs = len(corpus)
for token, postings in inverted_index.items():
    idf = num_docs / len(postings)
    for node in postings:
        node.tf_idf *= idf

# Add skip pointers
for token, postings in inverted_index.items():
    L = len(postings)
    skip_length = round(math.sqrt(L))
    if skip_length > 1:
        current = postings[0]
        count = 0
        for i in range(1, L):
            count += 1
            if count == skip_length:
                current.skip = postings[i]
                current = current.skip
                count = 0

def print_posting_lists(inverted_index):
    for term, postings in inverted_index.items():
        doc_ids = []
        current = postings[0]
        while current:
            doc_ids.append(current.doc_id)
            current = current.next
        print(f"{term} -> {', '.join(doc_ids)}")

print_posting_lists(inverted_index)
'''
        
        
import re
import math
from collections import defaultdict
from nltk.stem import PorterStemmer
from flask import Flask, request, jsonify

# Custom stop words list
stop_words = set(["i" ,"me", "my", "myself", "we", "our" ,"ours" ,"ourselves" ,"you" ,"your", "yours" ,"yourself" ,"yourselves", "he" ,"him" ,"his" ,"himself","she", "her", "hers" ,"herself" ,"it" ,"its" ,"itself" ,"they", "them" ,"their" ,"theirs" ,"themselves", "what", "which" ,"who" ,"whom", "this", "that", "these", "those", "am" ,"is", "are" ,"was", "were", "be" ,"been" ,"being" ,"have", "has" ,"had" ,"having", "do" ,"does", "did" ,"doing" ,"a" ,"an" ,"the" ,"and", "but" ,"if" ,"or", "because" ,"as" ,"until" ,"while", "of" ,"at" ,"by", "for" ,"with", "about", "against" ,"between", "into", "through" ,"during", "before", "after" ,"above", "below" ,"to" ,"from" ,"up" ,"down" ,"in" ,"out", "on" ,"off", "over", "under", "again", "further", "then", "once" ,"here", "there", "when", "where", "why" ,"how" ,"all", "any", "both" ,"each" ,"few" ,"more" ,"most", "other" ,"some", "such", "no" ,"nor" ,"not" ,"only", "own" ,"same", "so" ,"than" ,"too", "very", "s" ,"t" ,"can" ,"will", "just", "don" ,"should", "now"])

class ListNode:
    def __init__(self, doc_id, tf_idf):
        self.doc_id = doc_id
        self.tf_idf = tf_idf
        self.next = None
        self.skip = None


def process_document(document):
    # Convert to lowercase
    document = document.lower()
    # Remove special characters
    document = re.sub(r'[^a-z0-9 ]', '', document)
    # Remove extra spaces
    document = re.sub(r'\s+', ' ', document).strip()
    # Tokenize and remove stopwords
    tokens = [token for token in document.split() if token not in stop_words]
    # Apply stemming
    tokens = [stemmer.stem(token) for token in tokens]
    return tokens
# Load corpus
with open("input_corpus.txt", 'r') as f:
    corpus = [line.strip().split('\t') for line in f.readlines()]

# Initialize stemmer
stemmer = PorterStemmer()

# Build inverted index
inverted_index = defaultdict(list)
# ... your inverted index creation code here
for doc_id, document in corpus:
    tokens = process_document(document)
    doc_length = len(tokens)
    token_freqs = defaultdict(int)
    for token in tokens:
        token_freqs[token] += 1
    for token, freq in token_freqs.items():
        tf = freq / doc_length
        node = ListNode(doc_id, tf)
        # Ensure postings are stored in a linked list and ordered by doc_id
        if inverted_index[token]:
            last_node = inverted_index[token][-1]
            last_node.next = node
            inverted_index[token].append(node)
        else:
            inverted_index[token].append(node)

# Calculate IDF and TF-IDF
# ... your IDF and TF-IDF calculation code here
num_docs = len(corpus)
for token, postings in inverted_index.items():
    idf = num_docs / len(postings)
    for node in postings:
        node.tf_idf *= idf
# Add skip pointers
# ... your skip pointers code here
for token, postings in inverted_index.items():
    L = len(postings)
    skip_length = round(math.sqrt(L))
    if skip_length > 1:
        current = postings[0]
        count = 0
        for i in range(1, L):
            count += 1
            if count == skip_length:
                current.skip = postings[i]
                current = current.skip
                count = 0



def daat_and_query(tokens, inverted_index):
    tokens = [stemmer.stem(token) for token in tokens if token not in stop_words]
    if not tokens:
        return []
    sorted_tokens = sorted(tokens, key=lambda x: len(inverted_index.get(x, [])))
    if not inverted_index.get(sorted_tokens[0]):
        return []
    result = set([node.doc_id for node in inverted_index[sorted_tokens[0]]])
    for token in sorted_tokens[1:]:
        if not inverted_index.get(token):
            return []
        result &= set([node.doc_id for node in inverted_index[token]])
    return sorted(list(result))

def serialize_postings_list(postings):
    serialized_postings = []
    current = postings[0]
    while current:
        serialized_postings.append({
            "doc_id": current.doc_id,
            "tf_idf": current.tf_idf
        })
        current = current.next
    return serialized_postings

def serialize_inverted_index(inverted_index):
    serialized_index = {}
    for term, postings in inverted_index.items():
        serialized_index[term] = serialize_postings_list(postings)
    return serialized_index


app = Flask(__name__)

@app.route('/execute_query', methods=['POST'])
def execute_query():
    serialized_index = serialize_inverted_index(inverted_index)
    return jsonify(serialized_index)

if __name__ == "__main__":
    app.run(port=9999)









        
        
