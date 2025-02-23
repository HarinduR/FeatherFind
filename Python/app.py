from flask import Flask, request, jsonify
from FeatureExtractor import FeatureExtractor
from SPARQLQueryBuilder import SPARQLQueryBuilder
from OntologyQueryEngine import OntologyQueryEngine

app = Flask(__name__)

extractor = FeatureExtractor()
query_builder = SPARQLQueryBuilder()
ontology_engine = OntologyQueryEngine("C:/Users/Daham/Documents/GitHub/FeatherFind/Python/ontology.owl")

@app.route("/")
def home():
    return "FeatherFind Chatbot is running!"

@app.route("/query_bird", methods=["POST"])
def query_bird():
    data = request.get_json()
    text = data.get("text", "")
    
    features = extractor.extract_features(text)
    
    sparql_query = query_builder.build_query(features)
    
    results = ontology_engine.query(sparql_query)
    
    return jsonify({
        "query": sparql_query,
        "features": features,
        "results": results
    })

if __name__ == "__main__":
    app.run(debug=True,port=5001)