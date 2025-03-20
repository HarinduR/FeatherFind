import spacy
from spacy.matcher import Matcher
from spacy.util import filter_spans
from spacy.tokens import Span

class FeatureExtractor:

    def __init__(self):

        self.nlp = spacy.load("en_core_web_sm")

        self.COLOR_SYNONYMS = {
            "azure": "blue", "crimson": "red", "scarlet": "red",
            "emerald": "green", "ivory": "white", "charcoal": "black"
        }

        self.HABITAT_TERMS = ["forest", "wetland", "desert", "water", "mountains", "grassland", "marsh", "coast", "open field", "water body", "woodland"]
        self.SIZE_TERMS = ["small", "large", "tiny", "big", "medium", "giant"]
        self.CONTINENTS = ["asia", "europe", "africa", "america", "australia", "antarctica"]
        self.DIRECTIONS = ["north", "south", "east", "west"]
        self.BEAK_SHAPES = ["curved", "spare", "long", "short", "thick"]

    def extractFeatures(self, text):
        doc = self.nlp(text.lower())
        features = {
            "size": None,
            "color": {"primary": None, "secondary": None},
            "habitat": None,
            "region": None,
            "diet": None,
            "eyes": {"size": None, "color": None},
            "beak": {"size": None, "color": None},
            "legs": {"size": None, "color": None},
        }

        for token in doc:
            if token.dep_ == "amod" and token.head.text == "bird" and token.text in self.SIZE_TERMS:
                features["size"] = token.text

            if token.dep_ == "amod" and token.head.text in ["feathers", "wings", "chest", "body", "plumage", "tail"]:
                compound_color = None
                for child in token.head.children:
                    if child.dep_ == "compound":
                        compound_color = self.COLOR_SYNONYMS.get(child.text, child.text)
                        break

                color = compound_color if compound_color else self.COLOR_SYNONYMS.get(token.text, token.text)

                if not features["color"]["primary"]:
                    features["color"]["primary"] = color
                else:
                    features["color"]["secondary"] = color

            if token.dep_ == "amod" and token.head.text == "bird" and token.text not in self.SIZE_TERMS:
                color = self.COLOR_SYNONYMS.get(token.text, token.text)
                if not features["color"]["primary"]:
                    features["color"]["primary"] = color

            if token.dep_ == "amod" and token.children != "" and token.head.text in ["feathers", "wings", "chest", "body", "plumage", "tail", "bird"] and token.text not in self.SIZE_TERMS:
                for child in token.children:
                    if child.dep_ == "conj":
                        color = self.COLOR_SYNONYMS.get(child.text, child.text)
                        features['color']['secondary'] = color

        matcher = Matcher(self.nlp.vocab)  # Use self.nlp

        matcher.add("HABITAT", [
            [{"LOWER": {"IN": ["in", "near", "around", "found"]}},
            {"LOWER": {"IN": self.HABITAT_TERMS}}]
        ])

        matcher.add("REGION", [
            [
                {"LOWER": {"IN": ["in", "from", "found"]}},
                {"LOWER": {"IN": self.DIRECTIONS}, "OP": "*"},
                {"LOWER": {"IN": self.CONTINENTS}}
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

        matcher.add("BEAK_SHAPE", [
            [
                {"LOWER": {"IN": ["has", "with", "a", "had"]}},
                {"LOWER": {"IN": self.BEAK_SHAPES}},
                {"POS": {"IN": ["ADJ", "NUM"]}, "OP": "+"},
                {"LOWER": {"IN": ["beak", "bill"]}}
            ]
        ])

        matches = matcher(doc)
        spans = []

        if not Span.has_extension("match_label"):
            Span.set_extension("match_label", default=None)

        for match_id, start, end in matches:
            label = self.nlp.vocab.strings[match_id]
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
                if len(span) >= 2:
                    words = text.split()
                    category = words[-1]
                    for word in words[:-1]:
                        if word in self.SIZE_TERMS:
                            features[category]["size"] = word
                        elif word not in self.BEAK_SHAPES:
                            features[category]["color"] = self.COLOR_SYNONYMS.get(word, word)
            elif label == "BEAK_SHAPE":
                for word in text.split()[:-1]:
                    print(word)
                    if word in self.BEAK_SHAPES:
                        features["beak"]["size"] = word
                    else:
                        features["beak"]["color"] = self.COLOR_SYNONYMS.get(word, word)
        return features