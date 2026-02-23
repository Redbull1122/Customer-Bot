import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from src.rag_chain import generate_answer
from langchain_classic.memory import ConversationBufferWindowMemory

app = Flask(__name__)

# Get port from environment variable
PORT = int(os.environ.get('PORT', 8080))
CORS(app)  # This will enable CORS for all routes

# In-memory session management (for demonstration purposes)
sessions = {}
MAX_SESSIONS = int(os.environ.get("MAX_SESSIONS", "500"))


def _evict_old_sessions_if_needed() -> None:
    while len(sessions) > MAX_SESSIONS:
        oldest_session_id = next(iter(sessions))
        sessions.pop(oldest_session_id, None)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/status', methods=['GET'])
def status():
    return jsonify({
        "status": "online",
        "message": "Hajster Customer Bot API is running.",
        "endpoints": {
            "/chat": "POST, expects JSON with 'message' and optional 'session_id'"
        }
    })

@app.route('/favicon.ico')
def favicon():
    return '', 204

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    if request.method == 'GET':
        return jsonify({
            "message": "This endpoint requires a POST request with a 'message' in the JSON body.",
            "example": {
                "message": "Your query here",
                "session_id": "optional-session-id"
            }
        })

    data = request.json
    user_query = data.get('message')
    session_id = data.get('session_id', 'default') # Use a session_id to maintain conversation history

    if not user_query:
        return jsonify({"error": "Missing 'message' in request body"}), 400

    print(f"[LOG] Processing query: '{user_query}' for session: {session_id}")

    # Get or create a memory for the session
    if session_id not in sessions:
        sessions[session_id] = ConversationBufferWindowMemory(
            k=5,
            memory_key="history",
            return_messages=False
        )
        _evict_old_sessions_if_needed()
    else:
        # Refresh insertion order so active sessions are not evicted first
        sessions[session_id] = sessions.pop(session_id)
    
    memory = sessions[session_id]

    try:
        # Generate the answer using the RAG chain
        result = generate_answer(
            query_text=user_query,
            top_k=5,
            memory=memory,
            use_query_expansion=True,
            use_image_search=False,
            validate_answer=True
        )

        sources_count = len(result.get("sources", []))
        print(f"[LOG] Answer generated. Sources found: {sources_count}")
        metrics = result.get("debug_metrics", {})
        if metrics:
            print(
                f"[PERF][chat] retrieval={metrics.get('retrieval_ms')}ms "
                f"llm={metrics.get('llm_ms')}ms total={metrics.get('total_ms')}ms"
            )
        
        if sources_count == 0:
            print("[LOG] WARNING: No sources found! Check if Pinecone index is empty or query is irrelevant.")
        else:
            print(f"[LOG] Top source: {result['sources'][0].get('source', 'unknown')}")

        bot_response = {
            "answer": result.get("answer", ""),
            "sources": result.get("sources", [])
        }
        
        return jsonify(bot_response)

    except Exception as e:
        print(f"[ERROR] {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Cloud Run injects the PORT environment variable, so we use that.
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port, debug=True)
