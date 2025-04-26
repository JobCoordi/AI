import os
from typing import Sequence, List, Optional
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
                "당신은 직업 10년차 상담사입니다. 저의 질문을 듣고 제가 무슨 직업이 어울릴지 자세하게 설명해주세요."
            )

            self.prompt_template = ChatPromptTemplate.from_messages([
                ("system", system_prompt or default_prompt),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{question}"),
            ])

            self.memory = MemorySaver()
            self.workflow = self._build_workflow()
            self.app = self.workflow.compile(checkpointer=self.memory)

            self.chain = self.prompt_template | self.llm
            self.chain_with_history = RunnableWithMessageHistory(
                self.chain,
                self._get_chat_history,
                input_messages_key="question",
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

            last_human_message = next(
                (msg.content for msg in reversed(all_messages) if isinstance(msg, HumanMessage)),
                "거래처에 새해 인사와 함께 신규 계약건에 대해 이메일을 보내야 한다. 뭐라고 해야 할까?"
            )

            response = self.llm.invoke(
                self.prompt_template.format(
                    history=all_messages,
                    language=state.get("language", "korean"),
                    question=last_human_message
                )
            )

            return {
                "messages": all_messages + [response],
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
