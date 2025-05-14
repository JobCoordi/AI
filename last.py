import os
import logging
from typing import Optional, List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_message_histories import SQLChatMessageHistory

class FinalRecommendation:
    def __init__(self, model_name: str = "gpt-4o", db_connection: Optional[str] = None):
        # 로깅 설정
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO, 
                           format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # DB 연결 문자열 설정
        self.db_connection = db_connection or os.getenv("DB_CONNECTION_STRING")
        if not self.db_connection:
            self.logger.warning("DB_CONNECTION_STRING이 설정되지 않았습니다.")
            raise ValueError("데이터베이스 연결 문자열이 제공되지 않았습니다. db_connection 매개변수나 DB_CONNECTION_STRING 환경 변수를 설정하세요.")

        # LLM 모델 초기화
        try:
            self.llm = ChatOpenAI(model_name=model_name)
            self.logger.info(f"LLM 모델 '{model_name}' 초기화 완료")
        except Exception as e:
            self.logger.error(f"LLM 모델 초기화 오류: {e}")
            raise ValueError(f"LLM 모델 초기화 실패: {str(e)}")

        # 프롬프트 템플릿 설정
        self.final_prompt_template = ChatPromptTemplate.from_template(
            """다음은 진로 상담을 위한 사용자의 질문들입니다. 이를 기반으로 현실적인 직업 하나를 추천해 주세요.
직업명만 간단히 말해도 좋습니다. 설명은 왜 이 직업을 추천하고 어떠한 부분이 사용자가 이 직업에 잘 어울리는지 성격적으로나 학업쪽으로나 모든 부분을 고려해서서 작성하세요.
답변 형식은 아래와 같이 하세요 직업 대분류는 아래 리스트를 참고하세요 

[
    "사업관리",
    "경영·회계·사무",
    "금융·보험",
    "교육·자연·사회과학",
    "법률·경찰·소방·교도·국방",
    "보건·의료",
    "사회복지·종교",
    "문화·예술·디자인·방송",
    "운전·운송",
    "영업판매",
    "경비·청소",
    "이용·숙박·여행·오락·스포츠",
    "음식서비스",
    "건설",
    "기계",
    "재료",
    "화학·바이오(구.화학)",
    "섬유·의복",
    "전기·전자",
    "정보통신",
    "식품가공",
    "인쇄·목재·가구·공예",
    "환경·에너지·안전",
    "농림어업"
]

[직업:추천 이유:직업 대분류]

딱 필요한 말만 하고 설명 글이나 다른 말은 절대 하지 마세요
{messages}
"""
        )

    def _get_chat_history(self, session_id: str):
        """
        사용자 세션 ID를 기반으로 채팅 기록을 가져옵니다.
        연결 오류를 명시적으로 처리합니다.
        """
        try:
            return SQLChatMessageHistory(
                session_id=session_id, 
                connection_string=self.db_connection
            )
        except Exception as e:
            self.logger.error(f"채팅 기록 조회 오류 (session_id: {session_id}): {e}")
            raise ConnectionError(f"데이터베이스 연결 또는 채팅 기록 조회 오류: {str(e)}")

    def _extract_user_messages(self, messages: List[Any]) -> List[str]:
        """
        메시지 목록에서 사용자(human) 메시지만 추출합니다.
        여러 메시지 형식을 처리할 수 있도록 합니다.
        """
        user_inputs = []
        for msg in messages:
            # LangChain 메시지 객체 처리 방법 다양화
            if hasattr(msg, 'type') and msg.type == "human":
                user_inputs.append(msg.content)
            elif isinstance(msg, HumanMessage):
                user_inputs.append(msg.content)
            elif isinstance(msg, dict) and msg.get('type') == 'human':
                user_inputs.append(msg.get('content', ''))
            # 필요시 다른 형식도 추가
        
        return user_inputs

    def get_recommendation(self, user_id: str) -> str:
        """
        사용자 ID를 기반으로 채팅 기록을 분석하여 직업 추천을 제공합니다.
        다양한 오류 상황을 구체적으로 처리합니다.
        """
        try:
            # 채팅 기록 가져오기
            chat_history = self._get_chat_history(user_id)
            past_messages = chat_history.messages
            
            if not past_messages:
                self.logger.warning(f"사용자 {user_id}의 채팅 기록이 없습니다.")
                return "충분한 대화 기록이 없어 추천을 제공할 수 없습니다. 더 많은 정보를 공유해 주세요."
            
            # 사용자 메시지 추출
            user_inputs = self._extract_user_messages(past_messages)
            
            if not user_inputs:
                self.logger.warning(f"사용자 {user_id}의 메시지가 추출되지 않았습니다.")
                return "사용자 메시지를 찾을 수 없어 추천을 제공할 수 없습니다."
            
            # 메시지 결합 및 프롬프트 구성
            messages_combined = "\n".join(f"- {m}" for m in user_inputs)
            prompt = self.final_prompt_template.format(messages=messages_combined)
            
            # LLM 호출
            self.logger.info(f"사용자 {user_id}에 대한 직업 추천 생성 중...")
            response = self.llm.invoke(prompt)
            
            recommendation = response.content.strip()
            self.logger.info(f"사용자 {user_id}에 대한 직업 추천 완료: {recommendation}")
            
            return recommendation

        except ConnectionError as ce:
            self.logger.error(f"데이터베이스 연결 오류 (user_id: {user_id}): {ce}")
            return "데이터베이스 연결 문제로 추천을 제공할 수 없습니다. 나중에 다시 시도해 주세요."
            
        except ValueError as ve:
            self.logger.error(f"입력값 오류 (user_id: {user_id}): {ve}")
            return "입력 정보에 문제가 있어 추천을 제공할 수 없습니다."
            
        except Exception as e:
            self.logger.error(f"최종 추천 처리 중 예상치 못한 오류 (user_id: {user_id}): {e}")
            return "추천 결과를 생성할 수 없습니다. 시스템 오류가 발생했습니다."