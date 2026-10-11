#!/usr/bin/env python3
import json
import urllib.request

def main():
    url = "http://localhost:1234/v1/chat/completions"
    payload = {
        "model": "google/gemma-4-e4b",
        "messages": [
            {"role": "user", "content": "請用繁體中文寫一首關於人工智慧與程式設計的四行短詩。"}
        ],
        "max_tokens": 100,
        "temperature": 0.7
    }
    
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    
    print("發送測試請求至 vLLM Server (http://localhost:1234)...")
    try:
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode("utf-8"))
            content = result["choices"][0]["message"]["content"]
            tokens = result["usage"]["total_tokens"]
            print("\n=== 模型回覆 ===")
            print(content)
            print(f"\n[Token 統計] 總計使用: {tokens} tokens")
    except Exception as e:
        print(f"[錯誤] 請求失敗: {e}")

if __name__ == "__main__":
    main()
