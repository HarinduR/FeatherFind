from flask import Flask, request, jsonify
from flask_cors import CORS
from retrieve_answer import retrieve_answer  
from response_generator import generate_gpt2_response  

app = Flask(__name__)
CORS(app)

@app.route("/get_bird_info", methods=["POST"])
def get_bird_info():
    data = request.get_json()
    user_query = data.get("query")

    if not user_query:
        return jsonify({"response": "❌ No query provided. Please ask a bird-related question."}), 400

    retrieved_chunk = retrieve_answer(user_query)

    # ✅ If no relevant information found, return default response
    if "Sorry, I don't have information" in retrieved_chunk:
        return jsonify({"response": retrieved_chunk})

    if "tell me about" in user_query.lower():
        final_response = retrieved_chunk
    else:
        final_response = generate_gpt2_response(user_query, retrieved_chunk)

    return jsonify({"response": final_response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
