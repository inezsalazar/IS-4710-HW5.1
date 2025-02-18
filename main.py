#Original code taken from
#
#Author:
import spacy
import networkx as nx
import matplotlib.pyplot as plt
import nltk
from spacy_wordnet.wordnet_annotator import WordnetAnnotator

# Initialize spaCy model
nlp = spacy.load("en_core_web_trf")
nlp.add_pipe("spacy_wordnet", after="tagger")

def is_action_verb(token):
    if token.pos_ == "VERB":
        synsets = token._.wordnet.synsets()
        if not synsets:
            return False
        for synset in synsets:
            if synset.lexname() == 'verb.motion':
                lemmas =[lemma.name() for lemma in synset.lemmas()]
                if 'travel' in lemmas:
                    return 'go'
                else:
                    return True
            elif synset.lexname() == 'verb.communication':
                lemmas = [lemma.name() for lemma in synset.lemmas()]
                return 'say'

    return False

#    print(token)
#    print(token.dep_)
#   print(token.lemma)
#    print(token._.wordnet.synsets())
#    print(token._.wordnet.lemmas())
#    print(token._.wordnet.synsets()[0].lexname)
#    print(token._.wordnet.synsets()[0].lemma_names())

#print(is_action_verb(nlp("He went")[0]))



# Define the function for simple Semantic Role Labeling (SRL)
def simple_srl(sentence, nlp):
    doc = nlp(sentence)
    subjects = []
    verbs = []
    objects = []
    indirect_objects = []

    for token in doc:
        if "subj" in token.dep_:
            subjects.append(token.text)
        if "VERB" in token.pos_:
            action = is_action_verb(token)
            if type(action) == str:
                verbs.append(action)
            elif action:
                verbs.append(token.lemma_)

        if "obj" in token.dep_:
            objects.append(token.text)
        if "dative" in token.dep_:
            indirect_objects.append(token.text)

    return {
        'subjects': subjects,
        'verbs': verbs,
        'objects': objects,
        'indirect_objects': indirect_objects
    }


def build_and_plot_knowledge_graph_matplotlib(srl_results):
    G = nx.DiGraph()

    for result in srl_results:
        subjects = result['subjects']
        verbs = result['verbs']
        objects = result['objects']
        indirect_objects = result['indirect_objects']

        for subject in subjects:
            for verb in verbs:
                for obj in objects:
                    G.add_edge(subject, obj, label=verb)
                for ind_obj in indirect_objects:
                    G.add_edge(subject, ind_obj, label=verb)

    alice_decendants = list(G.successors("Alice"))
    subgraph_nodes = ["Alice"] + alice_decendants
    H = G.subgraph(subgraph_nodes).copy()

    pos = nx.spring_layout(H, seed=42)

    # Draw nodes and edges
    nx.draw(H, pos, with_labels=True, node_color="skyblue", node_size=2000, font_size=9, font_color="black",
            font_weight="bold", arrows=True)

    # Draw edge labels
    edge_labels = nx.get_edge_attributes(H, 'label')
    nx.draw_networkx_edge_labels(H, pos, edge_labels=edge_labels)

    # Show plot
    plt.show()

def load_text(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    return text

text = load_text('alice.txt')
# Process each sentence and extract SRL results
srl_results = []
for sent in nlp(text).sents:
    result = simple_srl(sent.text, nlp)
    srl_results.append(result)

print(srl_results)

# Build and plot the knowledge graph with matplotlib
build_and_plot_knowledge_graph_matplotlib(srl_results)

def query_graph(srl_results, subject_node,verb_edge):
    G = nx.DiGraph()

    for result in srl_results:
        subjects = result['subjects']
        verbs = result['verbs']
        objects = result['objects']
        indirect_objects = result['indirect_objects']

        for subject in subjects:
            for verb in verbs:
                for obj in objects:
                    G.add_edge(subject, obj, label=verb)
                for ind_obj in indirect_objects:

                    G.add_edge(subject, ind_obj, label=verb)
    answer = []
    edges = G.out_edges(subject_node)
    for u,v in edges:
        if G[u][v].get('label') == verb_edge:
            answer.append(v)
    return answer

query = "What did Alice say?"
for token in nlp(query):
    result = is_action_verb(token)
    if result == "go":
        #query graph for go edges
        print(query_graph(srl_results, "Alice", result))
        break
    elif result == "say":
        print(query_graph(srl_results, "Alice", result))
        break