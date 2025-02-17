import rdflib

class OntologyQueryEngine:

    def __init__(self, ontology_path):
        self.graph = rdflib.Graph()
        self.graph.parse(ontology_path, format="xml")

    def query(self, sparql_query):

        g = rdflib.Graph()
        g.parse("C:/Users/Daham/Documents/GitHub/FeatherFind/Chatbot/keyword-bird/ontology.owl", format="xml")
        results = g.query(sparql_query)
        
        output = []
        for row in results:
            output.append({
                "bird": str(row.bird),
                "commonName": str(row.commonName)
            })
        return output