import os
import re
import logging
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_community.llms import LlamaCpp
from langchain.prompts import PromptTemplate

# --- Конфигурация и Инициализация ---
logging.basicConfig(level=logging.INFO)
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
        n_ctx=2048, n_gpu_layers=25, verbose=False
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

    raw_output = raw_output.strip()
    logger.debug(f"Начинаем извлечение кода из:\n---\n{raw_output}\n---")


    code_block_match = re.search(r"```(?:rapid)?\s*(.*?)\s*```", raw_output, re.DOTALL | re.IGNORECASE)
    if code_block_match:
        code = code_block_match.group(1).strip()
        logger.info("Код извлечен из блока ```...```.")
        return code if code else None


    module_start_match = re.search(r"\bMODULE\b", raw_output, re.IGNORECASE)
    module_end_match = re.search(r"\bENDMODULE\b", raw_output, re.IGNORECASE)
    if module_start_match and module_end_match and module_end_match.end() > module_start_match.start():
        start_index = module_start_match.start()
        end_index = module_end_match.end()
        code = raw_output[start_index:end_index].strip()
        logger.info("Код извлечен из блока MODULE...ENDMODULE.")
        return code if code else None

        # 3. Поиск блока PROC main ... ENDPROC (менее надежно)
    proc_start_match = re.search(r"\bPROC\s+main\b", raw_output, re.IGNORECASE)
    proc_end_match = re.search(r"\bENDPROC\b", raw_output, re.IGNORECASE)
    if proc_start_match and proc_end_match and proc_end_match.end() > proc_start_match.start():

        end_index = proc_end_match.end()

        start_index = raw_output.rfind('\n', 0, proc_start_match.start()) + 1
        code = raw_output[start_index:end_index].strip()
        logger.warning("Извлечен только блок PROC main...ENDPROC (может быть неполным).")

        if not re.search(r"\bMODULE\b", code, re.IGNORECASE):
            code = f"MODULE MainModule\n    {code}\nENDMODULE"
        return code if code else None


    lines = raw_output.splitlines()
    code_lines = [
        line for line in lines
        if not re.match(r"^\s*(here's|вот|sure,|конечно|the code|is|:|\*).*", line, re.IGNORECASE)
    ]
    code = "\n".join(code_lines).strip()

    if code:
        logger.warning("Структурные блоки не найдены. Возвращается очищенный вывод.")
        # Если код все еще не похож на RAPID, возможно, вернуть None
        if not re.search(r"\b(MODULE|PROC|Move|SetDO|ResetDO|WaitTime)\b", code, re.IGNORECASE):
            logger.error("Не удалось извлечь валидный код после очистки.")
            return None
        return code
    else:
        logger.error("Не удалось извлечь код из вывода LLM (пустой результат после очистки).")
        return None


def validate_code(code: str) -> bool:

    # 1. Проверка на запрещенные команды (строго)
    forbidden_patterns = [r"\bConfJ\b", r"\bAccSet\b", r"\bz0\b", r"\bvmax\b"]
    if any(re.search(pattern, code, re.IGNORECASE) for pattern in forbidden_patterns):
        logger.warning("Код не прошел валидацию: Обнаружены запрещенные элементы.")
        return False

    # 2. Проверка наличия хотя бы одной разрешенной команды
    allowed_commands_pattern = r"\b(MoveL|MoveJ|MoveC|SetDO|ResetDO|WaitTime)\b"
    if not re.search(allowed_commands_pattern, code, re.IGNORECASE):
        logger.warning("Код не прошел валидацию: Не найдено ни одной разрешенной команды.")
        return False

    # 3. Базовая проверка структуры (желательно, но не строго обязательно для прохождения)
    has_module = re.search(r"\bMODULE\b", code, re.IGNORECASE) is not None
    has_proc_main = re.search(r"\bPROC\s+main\b", code, re.IGNORECASE) is not None
    has_endproc = re.search(r"\bENDPROC\b", code, re.IGNORECASE) is not None
    has_endmodule = re.search(r"\bENDMODULE\b", code, re.IGNORECASE) is not None

    if not (has_module and has_proc_main and has_endproc and has_endmodule):
        logger.warning(
            "Структура кода неполная (отсутствует MODULE/PROC main/ENDPROC/ENDMODULE), но проверка продолжается.")



    if re.search(r"\b(MoveL|MoveJ|MoveC)\b", code, re.IGNORECASE):
        if not re.search(r"\btool0\b", code, re.IGNORECASE):
            logger.warning("Код не прошел валидацию: Есть команда движения, но отсутствует 'tool0'.")
            return False

    # 5. Проверка точек с запятой в конце большинства строк (эвристика)
    lines = code.splitlines()
    relevant_lines = 0
    lines_with_semicolon = 0
    for line in lines:
        line = line.strip()
        if line and not line.startswith("!") and not line.startswith("MODULE") and not line.startswith(
                "ENDMODULE") and not line.startswith("PROC") and not line.startswith("ENDPROC"):
            relevant_lines += 1
            if line.endswith(";"):
                lines_with_semicolon += 1

    # Позволим небольшому числу строк (например, < 20%) быть без точки с запятой,
    # так как могут быть сложные структуры или ошибки парсинга
    if relevant_lines > 0 and (lines_with_semicolon / relevant_lines < 0.8):
        logger.warning(
            f"Код не прошел валидацию: Мало строк ({lines_with_semicolon}/{relevant_lines}) заканчиваются на ';'.")
        # return False # Можно сделать строже и вернуть False

    logger.info("Код прошел базовую валидацию.")
    return True


# --- Эндпоинт API ---
@app.post("/generate", summary="Генерирует RAPID код по описанию задачи")
async def generate_code(request: TaskRequest):
    """
    Принимает описание задачи и возвращает сгенерированный и валидированный RAPID код.
    """
    if llm is None or prompt is None:
        logger.error("LLM или Промпт не инициализированы.")
        raise HTTPException(status_code=503, detail="Сервис временно недоступен: ошибка инициализации.")

    try:
        logger.info(f"Получен запрос: '{request.task_description}'")
        formatted_prompt = prompt.format(task_description=request.task_description)

        logger.info("Вызов LLM...")
        raw_code = llm.invoke(formatted_prompt)
        logger.info("LLM вернула ответ.")

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


# --- Запуск Сервиса ---
if __name__ == "__main__":
    # Проверка перед запуском, что все инициализировалось
    if llm is None or prompt is None:
        logger.critical("Не удалось инициализировать LLM или Промпт. Сервер не может быть запущен.")
    else:
        import uvicorn

        logger.info("Запуск FastAPI приложения...")
        uvicorn.run(app, host="0.0.0.0", port=8000)



