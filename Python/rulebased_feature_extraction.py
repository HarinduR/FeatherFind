import spacy
from spacy.matcher import Matcher
from spacy.util import filter_spans
from spacy.tokens import Span

nlp = spacy.load("en_core_web_sm")

COLOR_SYNONYMS = {
    "azure": "blue", "crimson": "red", "scarlet": "red",
    "emerald": "green", "ivory": "white", "charcoal": "black"
}

HABITAT_TERMS = ["forest", "wetland", "desert", "water", "mountains", "grassland", "marsh", "coast"]
SIZE_TERMS = ["small", "large", "tiny", "big", "medium", "giant"]
CONTINENTS = ["asia", "europe", "africa", "america", "australia", "antarctica"]
DIRECTIONS = ["north", "south", "east", "west"]

def extract_features_optimized(text):
    doc = nlp(text.lower())
    features = {
        "size": None,
        "color": {"primary": None, "secondary": None},
        "habitat": None,
        "region": None,
        "diet": None,
        "eyes": {"size": None, "color": None},
        "beak": {"size": None, "color": None},
        "legs": {"size": None, "color": None},
        "feathers": None
    }

    for token in doc:
        if token.dep_ == "amod" and token.head.text == "bird" and token.text in SIZE_TERMS:
            features["size"] = token.text

        if token.dep_ == "amod" and token.head.text in ["feathers", "wings", "chest", "body", "plumage", "tail"]:
            compound_color = None
            for child in token.head.children:
                if child.dep_ == "compound":
                    compound_color = COLOR_SYNONYMS.get(child.text, child.text)
                    break

            color = compound_color if compound_color else COLOR_SYNONYMS.get(token.text, token.text)

            if not features["color"]["primary"]:
                features["color"]["primary"] = color
            else:
                features["color"]["secondary"] = color

        if token.dep_ == "amod" and token.head.text == "bird" and token.text not in SIZE_TERMS:
            color = COLOR_SYNONYMS.get(token.text, token.text)
            if not features["color"]["primary"]:
                features["color"]["primary"] = color
        
        if token.dep_ == "amod" and token.children != "" and token.head.text in ["feathers", "wings", "chest", "body", "plumage", "tail", "bird"] and token.text not in SIZE_TERMS:
          for child in token.children:
            if child.dep_ == "conj":
              color = COLOR_SYNONYMS.get(child.text, child.text)
              features['color']['secondary'] = color  

    matcher = Matcher(nlp.vocab)

    matcher.add("HABITAT", [
        [{"LOWER": {"IN": ["in", "near", "around", "found"]}},
         {"LOWER": {"IN": HABITAT_TERMS}}]
    ])

    matcher.add("REGION", [
        [
            {"LOWER": {"IN": ["in", "from", "found"]}},
            {"LOWER": {"IN": DIRECTIONS}, "OP": "*"},
            {"LOWER": {"IN": CONTINENTS}}
        ]
    ])

    matcher.add("DIET", [
        [{"LOWER": {"IN": ["eats", "feeds", "consumes", "diet"]}},
         {"POS": "NOUN"}]
    ])

    matcher.add("PHYSICAL_CHARACTERISTICS", [
        [
            {"POS": {"IN": ["ADV", "ADJ", "NUM"]}, "OP": "*"},
            {"POS": {"IN": ["ADJ", "NUM"]}, "OP": "+"},
            {"LOWER": {"IN": ["eyes", "beak", "bill", "legs"]}}
        ]
    ])

    matches = matcher(doc)
    spans = []

    if not Span.has_extension("match_label"):
        Span.set_extension("match_label", default=None)

    for match_id, start, end in matches:
        label = nlp.vocab.strings[match_id]
        span = doc[start:end]
        span._.match_label = label
        spans.append(span)

    filtered_spans = filter_spans(spans)

    for span in filtered_spans:
        label = span._.match_label
        text = span.text
        if label == "HABITAT":
            features["habitat"] = span[-1].text
        elif label == "REGION":
            features["region"] = span[1:].text
        elif label == "DIET":
            features["diet"] = span[-1].text
        elif label == "PHYSICAL_CHARACTERISTICS":
            if (len(span) >= 2):
              words = text.split()
              category = words[-1]
              for word in words[0:-1]:
                if word in SIZE_TERMS:
                  features[category]["size"] = word
                else:
                  features[category]["color"] = COLOR_SYNONYMS.get(word, word)

    return features

example_text = "giant bird with green body and azure feathers. black large beak. lives in water and eats crabs. this bird was seen in south asia. also it had  blue eyes "
example_text2 = "blue and green bird"
features = extract_features_optimized(example_text)
print(features)

def build_sparql_query(features):
    query = """
    PREFIX ex: <http://example.org/birds#>
    SELECT ?bird WHERE {
      ?bird a ex:Bird .
    """
    if features['color']['primary']:
        color_value = features['color']['primary'].capitalize()
        query += f'?bird ex:hasPrimaryColor ex:{color_value} .\n'
    if features['color']['secondary']:
        color_value = features['color']['secondary'].capitalize()
        query += f'?bird ex:hasSecondaryColor ex:{color_value} .\n'
    if features['habitat']:
        habitat_value = features['habitat'].capitalize()
        query += f'?bird ex:livesIn ex:{habitat_value} .\n'
    if features['region']:
        region_value = features['region'].replace(" ", "").capitalize()
        query += f'?bird ex:locatedIn ex:{region_value} .\n'
    query += "} LIMIT 5"
    return query

print(build_sparql_query(features))

import rdflib

def query_ontology(sparql_query):
    # Create an RDF graph
    g = rdflib.Graph()
    # Load your ontology file (adjust the filename and format as necessary)
    g.parse("C:/Users/Daham/Documents/GitHub/FeatherFind/Python/ontology-enhnaced.owx", format="xml")
    
    # Execute the SPARQL query
    results = g.query(sparql_query)
    
    # Process the results into a list of dictionaries, for example
    output = []
    for row in results:
        # Assuming your SELECT clause returns ?bird and ?commonName
        output.append({
            "bird": str(row.bird),
            "commonName": str(row.commonName)
        })
    return output

# Test the query function:
sparql_query = build_sparql_query(features)  # features from your extraction component
results = query_ontology(sparql_query)
print("Ontology Query Results:")
print(results)
