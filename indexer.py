#from linkedlist import LinkedList
from collections import OrderedDict
import math

class Indexer:
    def __init__(self):
        self.inverted_index = OrderedDict({})
        self.doc_lengths = {}  # For storing length of each document (count of terms)

    def get_index(self):
        return self.inverted_index

    def generate_inverted_index(self, doc_id, tokenized_document):
        for t in tokenized_document:
            self.add_to_index(t, doc_id)

    def add_to_index(self, term_, doc_id_):
        if term_ not in self.inverted_index:
            # If term doesn't exist, initialize it with a LinkedList
            self.inverted_index[term_] = LinkedList()

        # Add document ID to the term's linked list if it doesn't already exist
        if not self.inverted_index[term_].search(doc_id_):
            self.inverted_index[term_].append(doc_id_)

        # Update document length
        self.doc_lengths[doc_id_] = self.doc_lengths.get(doc_id_, 0) + 1

    def sort_terms(self):
        sorted_index = OrderedDict({})
        for k in sorted(self.inverted_index.keys()):
            sorted_index[k] = self.inverted_index[k]
        self.inverted_index = sorted_index

    def add_skip_connections(self):
        for _, postings_list in self.inverted_index.items():
            skip_step = int(math.sqrt(postings_list.size()))
            postings_list.add_skip_connections(skip_step)

    def calculate_tf_idf(self):
        total_docs = len(self.doc_lengths)

        for term, postings_list in self.inverted_index.items():
            doc_freq = postings_list.size()
            idf = math.log(total_docs / doc_freq)

            current = postings_list.start_node
            while current:
                tf = current.value[1]  # assuming (doc_id, term_freq) is stored in the list
                tf_idf_score = tf * idf
                # Modify the value to store (doc_id, tf-idf score)
                current.value = (current.value[0], tf_idf_score)
                current = current.next_node

