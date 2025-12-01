#!/usr/bin/env python3
"""
Simple test to verify Ollama Cloud provider integration
"""

import asyncio
from src.cgm_mcp.utils.config import LLMConfig
from src.cgm_mcp.utils.llm_client import LLMClient, OllamaCloudClient


def test_ollama_cloud_integration():
    """Test that the OllamaCloudClient is properly integrated"""
    
    print("🧪 Testing Ollama Cloud Integration...")
    
    # Test config creation
    config = LLMConfig(
        provider="ollama_cloud",
        model="llama3",
        api_key="test-key",
        api_base="https://api.ollama.example.com",
        temperature=0.1,
        max_tokens=200,
        timeout=30
    )
    
    print(f"✅ Config created with provider: {config.provider}")
    
    # Test direct client instantiation
    try:
        client = OllamaCloudClient(config)
        print("✅ OllamaCloudClient instantiated successfully")
    except Exception as e:
        print(f"❌ Error instantiating OllamaCloudClient: {e}")
        return False
    
    # Test main LLMClient with ollama_cloud provider
    try:
        llm_client = LLMClient(config)
        print("✅ Main LLMClient instantiated with ollama_cloud provider")
        
        # Check that the correct client type is created
        if isinstance(llm_client.client, OllamaCloudClient):
            print("✅ Main LLMClient correctly uses OllamaCloudClient")
        else:
            print(f"❌ Main LLMClient uses wrong client type: {type(llm_client.client)}")
            return False
    except Exception as e:
        print(f"❌ Error with main LLMClient: {e}")
        return False
    
    print("🎉 All integration tests passed!")
    return True


if __name__ == "__main__":
    test_ollama_cloud_integration()