import os
from typing import Sequence, Optional
import logging
from dotenv import load_dotenv

# PostgreSQL 드라이버
try:
    import psycopg2
except ImportError:
    logging.warning("psycopg2 not installed. Run: pip install psycopg2-binary")

from langchain_core.messages import HumanMessage, BaseMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

# Tavily 검색 도구
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import Tool, initialize_agent, AgentType

# 환경 변수 로드
load_dotenv()

class TranslationState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    language: str
    thread_id: str

class TranslationModel:
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        db_connection: Optional[str] = None,
        system_prompt: Optional[str] = None
    ):
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

        try:
            self.llm = ChatOpenAI(model_name=model_name)
            self.logger.info(f"초기화된 모델: {model_name}")

            self.db_connection = db_connection or os.getenv(
                "DB_CONNECTION_STRING",
                'postgresql://username:password@localhost:5432/chat_db'
            )

            default_prompt = (
                "당신은 경력 10년 이상의 진로 상담가입니다. 사용자의 정보를 보고 어떤 직업이 어울릴지 상담해주세요. "
                "이유도 구체적으로 설명하고, 어떤 준비가 필요한지도 조언해주세요."
            )

            self.prompt_template = ChatPromptTemplate.from_messages([
                ("system", system_prompt or default_prompt),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{profile_text}"),
            ])

            self.memory = MemorySaver()
            self.workflow = self._build_workflow()
            self.app = self.workflow.compile(checkpointer=self.memory)

            self.chain = self.prompt_template | self.llm
            self.chain_with_history = RunnableWithMessageHistory(
                self.chain,
                self._get_chat_history,
                input_messages_key="profile_text",
                history_messages_key="history",
            )

            self._test_db_connection()

        except Exception as e:
            self.logger.error(f"초기화 중 오류 발생: {str(e)}")
            raise

    def _test_db_connection(self):
        try:
            test_history = SQLChatMessageHistory(
                session_id="test_connection",
                connection_string=self.db_connection
            )
            self.logger.info("데이터베이스 연결 테스트: 성공")
        except Exception as e:
            self.logger.error(f"데이터베이스 연결 테스트 실패: {str(e)}")
            raise

    def _get_chat_history(self, session_id):
        return SQLChatMessageHistory(
            session_id=session_id,
            connection_string=self.db_connection
        )

    def _build_workflow(self):
        workflow = StateGraph(state_schema=TranslationState)
        workflow.add_node("model", self._call_model)
        workflow.add_edge(START, "model")
        return workflow

    def _call_model(self, state: TranslationState):
        try:
            thread_id = state["thread_id"]
            chat_history = self._get_chat_history(thread_id)

            past_messages = chat_history.messages
            current_messages = list(state["messages"])

            if past_messages and current_messages:
                last_message = current_messages[-1]
                if isinstance(last_message, HumanMessage):
                    all_messages = past_messages + [last_message]
                else:
                    all_messages = past_messages
            else:
                all_messages = current_messages or past_messages

            last_profile = next(
                (msg.content for msg in reversed(all_messages) if isinstance(msg, HumanMessage)),
                "출생년도:1999, 성별:남자, 최종학력:고졸, 전공:문과, 경력:없음, 관심분야:없음, 자격증:없음, 희망 근무 형태:없음, 자기소개:없음"
            )

            # Tavily 검색 도구 초기화
            search_tool = TavilySearchResults(api_key=os.getenv("TAVILY_API_KEY"))
            tools = [Tool.from_function(
                func=search_tool.run,
                name="TavilySearch",
                description="최신 직업 트렌드나 추천 정보를 검색합니다."
            )]

            # LLM + 툴 → 에이전트 구성
            agent_executor = initialize_agent(
                tools, self.llm,
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                verbose=False
            )

            # 실제 에이전트 실행
            response_text = agent_executor.run(
                f"{last_profile} 이 사용자에게 어울리는 최신 직업 트렌드와 준비 방법을 포함해서 조언해줘."
            )

            return {
                "messages": all_messages + [AIMessage(content=response_text)],
                "language": state.get("language", "korean"),
                "thread_id": thread_id
            }

        except Exception as e:
            self.logger.error(f"모델 호출 중 오류: {str(e)}")
            error_message = f"처리 중 오류가 발생했습니다: {str(e)}"
            return {
                "messages": (past_messages if 'past_messages' in locals() else []) + [AIMessage(content=error_message)],
                "language": state.get("language", "korean"),
                "thread_id": thread_id
            }

    def translate(self, text: str, session_id: str = "default_thread", language: str = "korean"):
        try:
            self.logger.info(f"세션 {session_id}에서 번역 요청 처리")

            chat_history = self._get_chat_history(session_id)
            past_messages = chat_history.messages
            user_message = HumanMessage(content=text)

            input_state = {
                "messages": past_messages + [user_message],
                "language": language,
                "thread_id": session_id
            }

            output = self.app.invoke(
                input_state,
                {"configurable": {"thread_id": session_id}}
            )

            all_messages = output["messages"]
            ai_response = all_messages[-1].content

            chat_history.add_user_message(text)
            chat_history.add_ai_message(ai_response)

            return ai_response

        except Exception as e:
            error_msg = f"번역 처리 중 오류: {str(e)}"
            self.logger.error(error_msg)
            return error_msg
