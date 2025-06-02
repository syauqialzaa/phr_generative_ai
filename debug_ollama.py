#!/usr/bin/env python3
"""
Debug script untuk testing koneksi Ollama
"""

import asyncio
import httpx
import json

OLLAMA_BASE_URL = "http://localhost:11434"

async def test_ollama_connection():
    """Test basic connection to Ollama"""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            if response.status_code == 200:
                data = response.json()
                print("✅ Ollama connection successful")
                print(f"📦 Available models: {len(data.get('models', []))}")
                for model in data.get('models', []):
                    print(f"   - {model.get('name', 'Unknown')}")
                return True
            else:
                print(f"❌ Ollama responded with status: {response.status_code}")
                return False
    except Exception as e:
        print(f"❌ Cannot connect to Ollama: {e}")
        return False

async def test_simple_chat():
    """Test simple chat with Ollama"""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Test simple generation first
            payload = {
                "model": "llama3.1",
                "prompt": "Say hello in one sentence.",
                "stream": False
            }
            
            print("\n🧪 Testing simple generation...")
            response = await client.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload)
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Generation successful: {data.get('response', '')[:100]}...")
                return True
            else:
                print(f"❌ Generation failed: {response.status_code}")
                print(f"Response: {response.text}")
                return False
                
    except Exception as e:
        print(f"❌ Generation test failed: {e}")
        return False

async def test_chat_api():
    """Test chat API with Ollama"""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Test chat API
            payload = {
                "model": "llama3.1",
                "messages": [
                    {"role": "user", "content": "Hello, say hi back in one sentence."}
                ],
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "num_predict": 50
                }
            }
            
            print("\n🧪 Testing chat API...")
            response = await client.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload)
            
            if response.status_code == 200:
                data = response.json()
                message = data.get('message', {}).get('content', '')
                print(f"✅ Chat successful: {message}")
                return True
            else:
                print(f"❌ Chat failed: {response.status_code}")
                print(f"Response: {response.text}")
                return False
                
    except Exception as e:
        print(f"❌ Chat test failed: {e}")
        return False

async def main():
    print("🔍 Debugging Ollama Connection")
    print("=" * 40)
    
    # Test 1: Basic connection
    print("\n1️⃣ Testing basic connection...")
    if not await test_ollama_connection():
        print("\n💡 Solutions:")
        print("1. Start Ollama: ollama serve")
        print("2. Check if port 11434 is free")
        print("3. Check firewall settings")
        return
    
    # Test 2: Simple generation
    print("\n2️⃣ Testing simple generation...")
    if not await test_simple_chat():
        print("\n💡 Solutions:")
        print("1. Install model: ollama pull llama3.1")
        print("2. Try different model: ollama pull llama3.2")
        print("3. Check model name: ollama list")
        return
    
    # Test 3: Chat API
    print("\n3️⃣ Testing chat API...")
    if not await test_chat_api():
        print("\n💡 Solutions:")
        print("1. Update Ollama to latest version")
        print("2. Restart Ollama service")
        return
    
    print("\n🎉 All tests passed! Ollama is working correctly.")
    print("\nYou can now run: python run_server.py")

if __name__ == "__main__":
    asyncio.run(main())