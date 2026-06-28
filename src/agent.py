"""
agent.py — Voice assistant for Al-Noor Clinic for Nutrition and Aesthetics.
Uses LiveKit Agents with Google Gemini and RAG from ChromaDB.
"""

from dotenv import load_dotenv
from livekit.agents import Agent, AgentSession, cli, WorkerOptions
from livekit.plugins import google
from rag_retriever import retrieve

load_dotenv()


from livekit.agents import llm

class ClinicKnowledgeBase(llm.Toolset):
    def __init__(self):
        super().__init__(id="clinic_db")

    @llm.function_tool(description="Search the clinic database to get accurate information about doctors, services, appointments, or prices.")
    async def search_clinic_info(self, query: str):
        print(f"[ClinicKnowledgeBase] Searching for: {query}")
        return retrieve(query)

class VoiceAssistant(Agent):
    def __init__(self) -> None:
        instructions = """You are a voice assistant for Al-Noor Clinic for Nutrition and Aesthetics in Benghazi.
Answer only based on the information available to you. Do not make up information.
If the patient asks you about doctors, services, prices, working hours, or any information about the clinic, you must use the tool (search_clinic_info) to search the database before answering.
If you do not find the answer in your information, tell the patient to contact the clinic directly.
Always speak in Arabic and in a friendly and professional tone.
"""
        super().__init__(
            instructions=instructions,
            llm=google.realtime.RealtimeModel(
                model="gemini-2.5-flash-native-audio-preview-12-2025",
                voice="Puck",
                temperature=0.8,
            ),
            tools=[ClinicKnowledgeBase()]
        )

async def entrypoint(ctx):
    await ctx.connect()
    
    session = AgentSession()
    await session.start(agent=VoiceAssistant(), room=ctx.room)

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))