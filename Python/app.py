from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/")
def get_bird():
    # Step 1: Get user input from the request
    data = request.json
    user_input = data.get("text", "")

    # Step 2: Extract features
    features = extract_features_optimized(user_input)

    # Step 3: Generate SPARQL query
    sparql_query = generate_sparql(features)

    # Step 4: Execute SPARQL query
    bird_names = execute_sparql(sparql_query)

    # Step 5: Return the result
    return jsonify({"birds": bird_names})


if __name__ == "__main__":
    app.run(debug=True)