import os
import logging
from typing import Sequence, Optional
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, BaseMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

load_dotenv()

class TranslationState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    language: str
    thread_id: str

class TranslationModel:
    def __init__(self, model_name: str = "gpt-4o", db_connection: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

        self.llm = ChatOpenAI(model_name=model_name)
        self.db_connection = db_connection or os.getenv("DB_CONNECTION_STRING")
        
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system",  """당신은 공감 능력이 뛰어난 전문 진로 상담가입니다. 
사용자의 학력, 경력, 기술 스택, 관심 분야, 성격, 고민 등을 바탕으로 현실적인 진로 방향을 제시하세요.

답변 시 다음 사항을 반드시 고려하세요:
1. 사용자의 말 속에 담긴 감정과 맥락을 잘 이해하고, 먼저 공감해 주세요.
2. 고민에 대해 구체적으로 어떤 선택지를 고려할 수 있을지 설명해 주세요.
3. 단순한 조언이 아니라, 사용자가 스스로 결정을 내릴 수 있도록 질문을 던지거나 방향을 제시해 주세요.
4. 사용자에게 도움이 될 수 있는 진로, 직무, 업계 정보 등을 현실적으로 예시로 들어 주세요.
5. 가능하다면 포트폴리오나 준비 방법, 공부 방법까지 구체적으로 제안해 주세요.

항상 진심 어린 태도로, 비판하지 않고 따뜻하게 조언해 주세요.
"""),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{text}")
        ])

        self.memory = MemorySaver()
        self.workflow = self._build_workflow()
        self.app = self.workflow.compile(checkpointer=self.memory)

        self.chain = self.prompt_template | self.llm
        self.chain_with_history = RunnableWithMessageHistory(
            self.chain,
            self._get_chat_history,
            input_messages_key="text",
            history_messages_key="history"
        )

        self._test_db_connection()

    def _test_db_connection(self):
        try:
            SQLChatMessageHistory(session_id="test", connection_string=self.db_connection)
            self.logger.info("✅ DB 연결 성공")
        except Exception as e:
            self.logger.error(f"❌ DB 연결 실패: {e}")
            raise

    def _get_chat_history(self, session_id):
        return SQLChatMessageHistory(session_id=session_id, connection_string=self.db_connection)

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

            all_messages = past_messages + current_messages

            last_user_message = next(
                (msg.content for msg in reversed(all_messages) if isinstance(msg, HumanMessage)),
                "사용자의 입력이 없습니다."
            )

            response = self.llm.invoke(
                self.prompt_template.format(history=all_messages, text=last_user_message)
            )

            return {
                "messages": all_messages + [response],
                "language": state.get("language", "korean"),
                "thread_id": thread_id
            }

        except Exception as e:
            error = f"모델 호출 실패: {e}"
            self.logger.error(error)
            return {
                "messages": all_messages + [AIMessage(content=error)],
                "language": state.get("language", "korean"),
                "thread_id": thread_id
            }

    def translate(self, user_id: str, content: str, language: str = "korean") -> str:
        try:
            chat_history = self._get_chat_history(user_id)
            past_messages = chat_history.messages
            user_message = HumanMessage(content=content)

            input_state = {
                "messages": past_messages + [user_message],
                "language": language,
                "thread_id": user_id
            }

            output = self.app.invoke(input_state, {"configurable": {"thread_id": user_id}})
            result = output["messages"][-1].content

            chat_history.add_user_message(content)
            chat_history.add_ai_message(result)

            return result

        except Exception as e:
            error = f"처리 중 오류: {e}"
            self.logger.error(error)
            return error
