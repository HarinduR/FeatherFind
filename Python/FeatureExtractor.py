import spacy
from spacy.matcher import Matcher
from spacy.util import filter_spans
from spacy.tokens import Span

class FeatureExtractor:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

        # Expanded color synonyms
        self.COLOR_SYNONYMS = {
            "azure": "blue", "crimson": "red", "scarlet": "red",
            "emerald": "green", "ivory": "white", "charcoal": "black",
            "golden": "yellow", "amber": "yellow", "sandy": "brown",
            "lavender": "purple", "grey": "gray", "turquoise": "blue",
            "magenta": "pink", "coral": "orange", "teal": "green",
            "maroon": "red", "burgundy": "red", "indigo": "blue"
        }

        # Expanded habitat terms
        self.HABITAT_TERMS = [
            "forest", "jungle", "wetland", "desert", "water", "mountains",
            "grassland", "marsh", "coast", "field", "woodland", "garden",
            "savanna", "river", "rainforest", "marshlands", "highlands", "lake",
            "scrubland", "prairies", "tundra", "alpine", "mangroves", "ocean",
            "outback", "grove", "orchard", "valley", "waterfall", "pond", "city", "suburban", "urban"
        ]

        # Expanded size terms
        self.SIZE_TERMS = ["small", "large", "tiny", "big", "medium", "giant", "massive", "huge", "curved", "spare", "long", "short", "thick", "hooked", "slender", "broad", "muscular"]

        self.CONTINENTS = ["asia", "europe", "africa", "america", "australia", "antarctica"]
        self.DIRECTIONS = ["north", "south", "east", "west", "central"]

        # Expanded beak shapes
        self.BEAK_SHAPES = ["curved", "spare", "long", "short", "thick", "hooked", "slender", "broad"]

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

        # Extract size and color features
        for token in doc:
            if token.dep_ == "amod" and token.head.text in ["bird", "body"] and token.text in self.SIZE_TERMS:
                features["size"] = token.text  # Store the size term

            # Color detection on feathers/body parts
            if token.dep_ == "amod" and token.head.text in ["feathers", "wings", "chest", "body", "plumage", "tail"]:
                if token.text not in self.SIZE_TERMS:  # Ensure it's not a size term
                    compound_color = None
                    for child in token.head.children:
                        if child.dep_ == "compound":
                            compound_color = self.COLOR_SYNONYMS.get(child.text, child.text)
                            break
                    color = compound_color if compound_color else self.COLOR_SYNONYMS.get(token.text, token.text)

                    # Assign colors correctly
                    if not features["color"]["primary"]:
                        features["color"]["primary"] = color
                    elif not features["color"]["secondary"] and color != features["color"]["primary"]:
                        features["color"]["secondary"] = color

                    # Handle conjunctions (e.g., "blue and green feathers")
                    for conj in token.conjuncts:
                        conj_color = self.COLOR_SYNONYMS.get(conj.text, conj.text)
                        if conj_color != features["color"]["primary"] and not features["color"]["secondary"]:
                            features["color"]["secondary"] = conj_color

            # Direct color modifiers on bird and body
            if token.dep_ == "amod" and token.head.text in ["bird", "body","feathers", "wings", "chest", "body", "plumage", "tail"]:
                if token.text not in self.SIZE_TERMS:  # Ensure it's not a size term
                    color = self.COLOR_SYNONYMS.get(token.text, token.text)

                    # Assign colors correctly
                    if not features["color"]["primary"]:
                        features["color"]["primary"] = color
                    elif not features["color"]["secondary"] and color != features["color"]["primary"]:
                        features["color"]["secondary"] = color

                    # Handle conjunctions (e.g., "blue and green bird")
                    for conj in token.conjuncts:
                        conj_color = self.COLOR_SYNONYMS.get(conj.text, conj.text)
                        if conj_color != features["color"]["primary"] and not features["color"]["secondary"]:
                            features["color"]["secondary"] = conj_color

            # Body part feature extraction
            if token.text in ["eyes", "beak", "bill", "legs"]:
                category = "beak" if token.text == "bill" else token.text
                adjectives = []

                # Direct modifiers and compounds
                for child in token.children:
                    if child.dep_ in ("amod", "compound"):
                        if child.text in self.COLOR_SYNONYMS:
                            adj = self.COLOR_SYNONYMS[child.text]
                            adjectives.append(adj)
                        elif child.text in self.SIZE_TERMS:
                            adjectives.append(child.text)

                # Conjunction handling (e.g., "small and black")
                adjectives = []  # Preserve order
                seen = set()  # Track unique adjectives to prevent duplicates

                for child in token.children:
                    if child.dep_ == "amod":
                        # Process the main adjective
                        adj_text = child.text.lower()
                        mapped_text = self.COLOR_SYNONYMS.get(adj_text, adj_text)  # Map colors

                        if mapped_text in self.SIZE_TERMS or mapped_text in self.COLOR_SYNONYMS.values() or mapped_text in self.BEAK_SHAPES:
                            if mapped_text not in seen:  # Avoid duplicates
                                adjectives.append(mapped_text)
                                seen.add(mapped_text)

                        # Process conjunctions
                        for conj in child.conjuncts:
                            conj_text = conj.text.lower()
                            mapped_conj_text = self.COLOR_SYNONYMS.get(conj_text, conj_text)

                            if mapped_conj_text in self.SIZE_TERMS or mapped_conj_text in self.COLOR_SYNONYMS.values() or self.BEAK_SHAPES:
                                if mapped_conj_text not in seen:
                                    adjectives.append(mapped_conj_text)
                                    seen.add(mapped_conj_text)  # Preserve unknown adjectives


                # Final assignment
                colors = [a for a in adjectives if a in self.COLOR_SYNONYMS.values()]
                sizes = [a for a in adjectives if a in self.SIZE_TERMS]

                if colors:
                    features[category]["color"] = colors[0] if len(colors) == 1 else colors
                if sizes:
                    features[category]["size"] = sizes[0] if len(sizes) == 1 else sizes

        # Initialize matcher
        matcher = Matcher(self.nlp.vocab)

        # Habitat pattern (now handles multi-word habitats)
        matcher.add("HABITAT", [
            [{"LOWER": {"IN": ["in", "near", "around", "along", "across"]}},
             {"IS_STOP": True, "OP": "*"},
             {"LOWER": {"IN": self.HABITAT_TERMS}}]
        ])

        # Region pattern (captures full region name)
        matcher.add("REGION", [
            [{"LOWER": {"IN": ["in", "from", "found", "of"]}},
             {"LOWER": {"IN": self.DIRECTIONS}, "OP": "*"},
             {"LOWER": {"IN": self.CONTINENTS}}]
        ])

        # Diet pattern
        matcher.add("DIET", [
            [{"LOWER": {"IN": ["eats", "feeds", "consumes", "diet", "hunts", "eating", "sipping", "preys"]}},
             {"IS_STOP": True, "OP": "*"},
             {"POS": {"IN": ["NOUN", "PROPN"]}}]
        ])

        # Physical characteristics pattern
        matcher.add("PHYSICAL_CHARACTERISTICS", [
            [{"POS": {"IN": ["ADV", "ADJ", "NUM"]}, "OP": "*"},
             {"POS": {"IN": ["ADJ", "NUM", "NOUN"]}, "OP": "+"},
             {"LOWER": {"IN": ["eyes", "beak", "bill", "legs"]}}]
        ])

        # Beak shape pattern
        matcher.add("BEAK_SHAPE", [
            [{"LOWER": {"IN": self.BEAK_SHAPES}},
             {"POS": {"IN": ["ADJ", "NUM", "NOUN"]}, "OP": "*"},
             {"LOWER": {"IN": ["beak", "bill"]}}],
            [{"POS": {"IN": ["ADJ", "NUM", "NOUN"]}, "OP": "*"},
             {"LOWER": {"IN": self.BEAK_SHAPES}},
             {"LOWER": {"IN": ["beak", "bill"]}}]
        ])

        # Match and process spans
        matches = matcher(doc)
        spans = []
        Span.set_extension("match_label", default=None, force=True)

        for match_id, start, end in matches:
            label = self.nlp.vocab.strings[match_id]
            span = doc[start:end]
            span._.match_label = label
            spans.append(span)

        filtered_spans = filter_spans(spans)

        for span in filtered_spans:
            label = span._.match_label
            if label == "HABITAT":
                features["habitat"] = span[-1].text  # Get last token of habitat span
            elif label == "REGION":
                # Join all tokens after preposition for full region name
                features["region"] = " ".join([token.text for token in span[1:]])
            elif label == "DIET":
                features["diet"] = span[-1].text
            elif label == "PHYSICAL_CHARACTERISTICS":
                category = span[-1].text
                category = "beak" if category == "bill" else category  # Handle "bill" as "beak"

                for token in span:
                    # Extract size terms
                    if token.text in self.SIZE_TERMS:
                        features[category]["size"] = token.text

                    # Extract colors (with synonyms)
                    elif token.text in self.COLOR_SYNONYMS:
                        features[category]["color"] = self.COLOR_SYNONYMS.get(token.text, token.text)

                    # Special handling for compound colors (e.g., "dark brown eyes")
                    elif token.dep_ == "amod" and token.head.text == category:
                        compound_color = None
                        for child in token.head.children:
                            if child.dep_ == "compound":
                                compound_color = self.COLOR_SYNONYMS.get(child.text, child.text)
                                break
                        color = compound_color if compound_color else self.COLOR_SYNONYMS.get(token.text, token.text)
                        features[category]["color"] = color

            elif label == "BEAK_SHAPE":
                category = "beak"
                for token in span:
                    if token.text in self.BEAK_SHAPES:
                        features[category]["size"] = token.text
                    elif token.text in self.COLOR_SYNONYMS:
                        features[category]["color"] = self.COLOR_SYNONYMS.get(token.text, token.text)

        return features