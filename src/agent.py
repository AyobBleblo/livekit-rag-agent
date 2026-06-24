"""
agent.py — المساعد الصوتي لعيادة النور للتغذية والتجميل.
يستخدم LiveKit Agents مع Google Gemini و RAG من ChromaDB.
"""

from dotenv import load_dotenv
from livekit.agents import Agent, AgentSession, cli, WorkerOptions
from livekit.plugins import google
from rag_retriever import retrieve
import asyncio

load_dotenv()


from livekit.agents import llm

class ClinicKnowledgeBase(llm.Toolset):
    def __init__(self):
        super().__init__(id="clinic_db")

    @llm.function_tool(description="ابحث في قاعدة بيانات العيادة للحصول على معلومات دقيقة عن الأطباء، الخدمات، المواعيد، أو الأسعار.")
    async def search_clinic_info(self, query: str):
        print(f"[ClinicKnowledgeBase] Searching for: {query}")
        return retrieve(query)

class VoiceAssistant(Agent):
    def __init__(self, initial_context: str) -> None:
        instructions = f"""أنت مساعد صوتي لعيادة النور للتغذية والتجميل في بنغازي.
أجب فقط بناءً على المعلومات المتوفرة لك. ولا تخترع معلومات من عندك.
إذا سألك المريض عن الأطباء أو الخدمات أو الأسعار، يجب عليك استخدام أداة (search_clinic_info) للبحث في قاعدة البيانات قبل الإجابة.
إذا لم تجد الإجابة في معلوماتك، قل للمريض أن يتواصل مباشرة مع العيادة.
تكلم بالعربية دائماً وبلهجة ودودة ومحترفة.

=== معلومات أساسية عن العيادة ===
{initial_context}
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
    
    # Load basic clinic info for the system prompt
    initial_context = retrieve("معلومات العيادة العامة ومواعيد العمل")
    
    session = AgentSession()
    await session.start(agent=VoiceAssistant(initial_context), room=ctx.room)

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))