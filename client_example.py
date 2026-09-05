"""Example client script to interact with the running Multi-Agent Analyst API.

Connects to the WebSocket streaming endpoint to audit `tests/sample_code.py`.
"""

import asyncio
import json
import sys
import httpx
import websockets


async def run_demo():
    api_base = "http://localhost:8000"
    ws_uri = "ws://localhost:8000/ws/analyze"

    print("==================================================")
    print("      Multi-Agent Analyst - Quick Demo Client     ")
    print("==================================================")

    # 1. Check API Health
    print("\n[1/3] Checking API Server Health...")
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = client.get(f"{api_base}/health")
            if resp.status_code == 200:
                print(f"  [OK] API is healthy! Details: {resp.json()}")
            else:
                print(f"  [!] API returned status code {resp.status_code}")
                return
    except Exception as exc:
        print(f"  [ERROR] Cannot reach API at {api_base}: {exc}")
        print("  --> Please ensure the API is running with: .\\run.ps1 or python app/main.py")
        return

    # 2. Check Models
    print("\n[2/3] Checking Active Models...")
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            models_resp = client.get(f"{api_base}/api/v1/models")
            print(f"  Active configuration: {models_resp.json()}")
    except Exception as exc:
        print(f"  [WARNING] Could not fetch models list: {exc}")

    # 3. Stream Analysis via WebSocket
    print("\n[3/3] Streaming Code Audit via WebSocket...")
    payload = {
        "query": "Please audit tests/sample_code.py. Inspect its functions, classes, and recommend optimizations.",
        "thread_id": "demo_session_001",
        "target_file": "tests/sample_code.py",
    }

    try:
        async with websockets.connect(ws_uri) as ws:
            print(f"  Connected to {ws_uri}")
            print(f"  Sending prompt: '{payload['query']}'\n")
            await ws.send(json.dumps(payload))

            while True:
                msg = await ws.recv()
                data = json.loads(msg)

                if "error" in data:
                    print(f"\n[ERROR]: {data['error']}")
                    break

                node = data.get("node")
                next_node = data.get("next_node")
                latest_message = data.get("latest_message")
                status = data.get("status")

                if node:
                    print(f"--> [Node Executing]: {node}")
                    if next_node:
                        print(f"    Next Routing: {next_node}")
                    if latest_message:
                        print(f"    Message Output:\n{latest_message}\n")

                if status == "completed":
                    print("\n[SUCCESS] Workflow completed successfully!")
                    break

    except Exception as exc:
        print(f"  [ERROR] WebSocket error: {exc}")


if __name__ == "__main__":
    asyncio.run(run_demo())
