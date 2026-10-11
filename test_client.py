#!/usr/bin/env python3
import json
import urllib.request

def main():
    url = "http://localhost:1234/v1/chat/completions"
    payload = {
        "model": "google/gemma-4-e4b",
        "messages": [
            {"role": "user", "content": "hello, world"}
        ],
        "max_tokens": 100,
        "temperature": 0.7
    }
    
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    
    print("Sending test request to vLLM Server (http://localhost:1234)...")
    try:
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode("utf-8"))
            content = result["choices"][0]["message"]["content"]
            tokens = result["usage"]["total_tokens"]
            print("\n=== Model Response ===")
            print(content)
            print(f"\n[Token Usage] Total: {tokens} tokens")
    except Exception as e:
        print(f"[Error] Request failed: {e}")

if __name__ == "__main__":
    main()
