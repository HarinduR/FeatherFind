from flask import Flask, request, jsonify
from FeatureExtractor import FeatureExtractor
from SPARQLQueryBuilder import SPARQLQueryBuilder
from OntologyQueryEngine import OntologyQueryEngine

app = Flask(__name__)

extractor = FeatureExtractor()
query_builder = SPARQLQueryBuilder()
ontology_engine = OntologyQueryEngine()

@app.route("/")
def home():
    return "FeatherFind Chatbot is running!"

@app.route("/query_bird", methods=["POST"])
def query_bird():
    data = request.get_json()
    text = data.get("text", "")
    
    features = extractor.extractFeatures(text)
    
    sparql_query = query_builder.build_query(features)
    
    results = ontology_engine.query(sparql_query)

    final_list = []

    for item in results:
        final_list.append(item.values())
    
    return jsonify({
        "features": features,
        "results": results
    })

if __name__ == "__main__":
    app.run(debug=True, port=5001)