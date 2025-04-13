import os
import re
import logging
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_community.llms import LlamaCpp
from langchain.prompts import PromptTemplate

# --- Конфигурация и Инициализация ---
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAPID Code Generator",
    description="API для генерации RAPID кода с использованием локальной LLM",
    version="1.1.0"
)

MODEL_PATH = "G:/LM/Models/lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
PROMPT_FILENAME = "rapid_prompt.txt"
SCRIPT_DIR = Path(__file__).parent
PROMPT_PATH = SCRIPT_DIR / "prompts" / PROMPT_FILENAME

llm = None
prompt = None
try:
    if not Path(MODEL_PATH).is_file():
        raise FileNotFoundError(f"Файл модели не найден: {MODEL_PATH}")
    llm = LlamaCpp(
        model_path=MODEL_PATH, temperature=0.3, max_tokens=1000,
        n_ctx=4090, n_gpu_layers=15, verbose=False
    )
    logger.info("LLM инициализирована успешно.")

    if not PROMPT_PATH.is_file():
         raise FileNotFoundError(f"Файл промпта не найден: {PROMPT_PATH}")
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        template = f.read()
    prompt = PromptTemplate.from_template(template)
    logger.info("Промпт успешно загружен.")

except Exception as e:
    logger.exception(f"Критическая ошибка при инициализации: {e}")




class TaskRequest(BaseModel):
    task_description: str


def extract_rapid_code(raw_output: str) -> str | None:
    """
    Извлекает ПЕРВЫЙ найденный блок кода RAPID (либо ```...```, либо MODULE...ENDMODULE).
    Отсекает ВЕСЬ текст ДО начала блока и ВЕСЬ текст ПОСЛЕ конца найденного блока.
    """
    raw_output = raw_output.strip()
    # logger.debug(f"Начинаем извлечение кода из:\n---\n{raw_output}\n---") # Можно оставить для отладки
    code = None

    # 1. Поиск ПЕРВОГО блока ```rapid ... ``` или ``` ... ```
    # Используем НЕЖАДНЫЙ поиск (.*?), чтобы найти самый короткий блок, если их несколько
    code_block_match = re.search(r"```(?:rapid)?\s*(.*?)\s*```", raw_output, re.DOTALL | re.IGNORECASE)
    if code_block_match:
        code = code_block_match.group(1).strip()
        logger.info("Код извлечен из ПЕРВОГО блока ```...```.")
        return code if code else None # Возвращаем сразу

    # 2. Если не нашли ```, ищем ПЕРВЫЙ блок MODULE ... ENDMODULE
    # Ищем самое первое вхождение MODULE
    module_start_match = re.search(r"\bMODULE\b", raw_output, re.IGNORECASE)
    if module_start_match:
        start_index = module_start_match.start()
        # Ищем самый первый ENDMODULE *после* найденного MODULE
        module_end_match = re.search(r"\bENDMODULE\b", raw_output[start_index:], re.IGNORECASE)
        if module_end_match:
            # Конец блока ENDMODULE относительно НАЧАЛА ИСХОДНОЙ СТРОКИ raw_output
            end_index = start_index + module_end_match.end()
            # Извлекаем ТОЛЬКО этот блок
            code = raw_output[start_index:end_index].strip()
            logger.info("Код извлечен из ПЕРВОГО блока MODULE...ENDMODULE.")
            return code if code else None # Возвращаем сразу
        else:
            logger.warning("Найден MODULE, но не найден ENDMODULE после него.")
    else:
        logger.warning("Структурный блок MODULE не найден.")

    # 3. Если не найдены стандартные блоки, возвращаем None
    logger.error("Не удалось извлечь структурный блок кода (``` или MODULE...ENDMODULE).")
    return None


def validate_code(code: str) -> bool:

    # Базовая проверка на пустоту
    if not code or not code.strip():
        logger.warning("Валидация не пройдена: Получен пустой код.")
        return False

    # 1. Проверка на запрещенные команды (строго) - ОСТАВЛЯЕМ
    forbidden_patterns = [r"\bConfJ\b", r"\bAccSet\b", r"\bz0\b", r"\bvmax\b"] # Добавьте/измените по необходимости
    if any(re.search(pattern, code, re.IGNORECASE) for pattern in forbidden_patterns):
        logger.warning("Код не прошел валидацию: Обнаружены ЗАПРЕЩЕННЫЕ элементы.")
        return False

    # 2. Проверка наличия хотя бы одной разрешенной команды - ОСТАВЛЯЕМ
    allowed_commands_pattern = r"\b(MoveL|MoveJ|MoveC|SetDO|ResetDO|WaitTime)\b"
    if not re.search(allowed_commands_pattern, code, re.IGNORECASE):
        logger.warning("Код не прошел валидацию: Не найдено ни одной РАЗРЕШЕННОЙ команды.")
        return False

    # ---------------------------------------------------------------
    # ---------------------------------------------------------------
    # # 3. Базовая проверка структуры (желательно, но не строго обязательно для прохождения)
    # has_module = re.search(r"\bMODULE\b", code, re.IGNORECASE) is not None
    # has_proc_main = re.search(r"\bPROC\s+main\b", code, re.IGNORECASE) is not None
    # has_endproc = re.search(r"\bENDPROC\b", code, re.IGNORECASE) is not None
    # has_endmodule = re.search(r"\bENDMODULE\b", code, re.IGNORECASE) is not None
    #
    # if not (has_module and has_proc_main and has_endproc and has_endmodule):
    #     logger.warning(
    #         "Структура кода неполная (отсутствует MODULE/PROC main/ENDPROC/ENDMODULE), но проверка продолжается.")

    # # 4. Проверка наличия tool0, если есть команды движения
    # if re.search(r"\b(MoveL|MoveJ|MoveC)\b", code, re.IGNORECASE):
    #     if not re.search(r"\btool0\b", code, re.IGNORECASE):
    #         logger.warning("Код не прошел валидацию: Есть команда движения, но отсутствует 'tool0'.")
    #         return False # <-- Закомментировано

    # # 5. Проверка точек с запятой в конце большинства строк (эвристика)
    # lines = code.splitlines()
    # relevant_lines = 0
    # lines_with_semicolon = 0
    # for line in lines:
    #     line = line.strip()
    #     if line and not line.startswith("!") and not line.startswith("MODULE") and not line.startswith(
    #             "ENDMODULE") and not line.startswith("PROC") and not line.startswith("ENDPROC"):
    #         relevant_lines += 1
    #         if line.endswith(";"):
    #             lines_with_semicolon += 1
    #
    # # Позволим небольшому числу строк (например, < 20%) быть без точки с запятой,
    # # так как могут быть сложные структуры или ошибки парсинга
    # if relevant_lines > 0 and (lines_with_semicolon / relevant_lines < 0.8):
    #     logger.warning(
    #         f"Код не прошел валидацию: Мало строк ({lines_with_semicolon}/{relevant_lines}) заканчиваются на ';'.")
    #     # return False # <-- Закомментировано

# --- Эндпоинт API ---
@app.post("/generate", summary="Генерирует RAPID код по описанию задачи")
async def generate_code(request: TaskRequest):

    if llm is None or prompt is None:
        logger.error("LLM или Промпт не инициализированы.")
        raise HTTPException(status_code=503, detail="Сервис временно недоступен: ошибка инициализации.")

    try:
        logger.info(f"Получен запрос: '{request.task_description}'")
        formatted_prompt = prompt.format(task_description=request.task_description)

        logger.info("Вызов LLM...")
        raw_code = llm.invoke(formatted_prompt)
        logger.info(f"ВЫВОД от LLM:\n---\n{raw_code}\n---")

        code = extract_rapid_code(raw_code)
        if code is None:
            logger.error("Не удалось извлечь код из вывода LLM.")
            raise HTTPException(status_code=500, detail="Ошибка обработки вывода модели: не удалось извлечь код.")
        logger.debug(f"Извлеченный код для валидации:\n{code}")

        if not validate_code(code):
            # Используем 422 Unprocessable Entity, т.к. проблема в семантике сгенер. данных
            raise HTTPException(status_code=422, detail="Сгенерированный код не прошел базовую проверку.")

        logger.info("Генерация и валидация кода успешно завершены.")
        return {"generated_code": code}

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.exception(f"Неожиданная ошибка во время генерации кода: {e}")
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка сервера: {e}")



if __name__ == "__main__":
    # Проверка перед запуском, что все инициализировалось
    if llm is None or prompt is None:
        logger.critical("Не удалось инициализировать LLM или Промпт. Сервер не может быть запущен.")
    else:
        import uvicorn

        logger.info("Запуск FastAPI приложения...")
        uvicorn.run(app, host="0.0.0.0", port=8000)



