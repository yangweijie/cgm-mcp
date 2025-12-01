#!/usr/bin/env python3
"""
Test script for Ollama Cloud provider
"""

import asyncio
import os
from src.cgm_mcp.utils.config import LLMConfig
from src.cgm_mcp.utils.llm_client import LLMClient, OllamaCloudClient


async def test_ollama_cloud_client():
    """Test the Ollama Cloud client functionality"""
    
    print("🧪 Testing Ollama Cloud Client...")
    
    # Create config for Ollama Cloud
    config = LLMConfig(
        provider="ollama_cloud",
        model=os.getenv("CGM_OLLAMA_CLOUD_MODEL", "llama3"),  # Default model for Ollama Cloud
        api_key=os.getenv("CGM_OLLAMA_CLOUD_API_KEY") or os.getenv("CGM_LLM_API_KEY", "your-ollama-cloud-api-key"),
        api_base=os.getenv("CGM_OLLAMA_CLOUD_API_BASE", "https://api.ollama.ai/v1"),  # Example Ollama Cloud endpoint
        temperature=0.1,
        max_tokens=500,
        timeout=60
    )
    
    print(f"Using model: {config.model}")
    print(f"Using API base: {config.api_base}")
    
    # Test direct client instantiation
    try:
        client = OllamaCloudClient(config)
        print("✅ OllamaCloudClient instantiated successfully")
        
        # Test health check
        health_status = await client.health_check()
        print(f"🏥 Health check status: {health_status}")
        
        if health_status:
            # Test generation
            print("📝 Testing text generation...")
            prompt = "Hello, this is a test. Please respond with a simple 'Hello, World!' and nothing else."
            response = await client.generate(prompt)
            print(f"💬 Response: {response}")
        else:
            print("⚠️  Health check failed - API key or endpoint may be incorrect")
            
    except Exception as e:
        print(f"❌ Error with direct client: {e}")
    
    # Test through main LLMClient
    try:
        print("\n🧪 Testing through main LLMClient...")
        llm_client = LLMClient(config)
        print("✅ Main LLMClient instantiated successfully with ollama_cloud provider")
        
        # Test health check
        health_status = await llm_client.health_check()
        print(f"🏥 Main client health check status: {health_status}")
        
        if health_status:
            # Test generation
            print("📝 Testing text generation through main client...")
            prompt = "Hello, this is a test. Please respond with 'Hello from Ollama Cloud!' and nothing else."
            response = await llm_client.generate(prompt)
            print(f"💬 Response: {response}")
        else:
            print("⚠️  Health check failed for main client")
            
    except Exception as e:
        print(f"❌ Error with main LLMClient: {e}")


if __name__ == "__main__":
    asyncio.run(test_ollama_cloud_client())