import rdflib

class OntologyQueryEngine:

    def query(self, sparql_query):

        g = rdflib.Graph()
        g.parse("ontology.owl", format="xml")
        results = g.query(sparql_query)
        
        output = []
        for row in results:
            full_uri = str(row.bird)
            bird_name = full_uri.split("#")[-1]  
            
            bird_name = bird_name.replace("_", " ")  
            
            output.append({"bird": bird_name})
        return output