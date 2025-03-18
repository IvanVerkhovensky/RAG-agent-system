from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
import re

app = FastAPI()

MODEL_PATH = "../models/llama-3-8b.Q4_K_M.gguf"

llm = LlamaCpp(
    model_path=MODEL_PATH,
    temperature=0.3,
    max_tokens=1000,
    n_ctx=2048,
    n_gpu_layers=40,  # Для GPU
    verbose=False
)

with open("app/prompts/rapid_prompt.txt", "r", encoding="utf-8") as f:
    template = f.read()

prompt = PromptTemplate.from_template(template)


class TaskRequest(BaseModel):
    task_description: str


def validate_code(code: str) -> bool:
    """Проверка безопасности кода"""
    required = [
        "MODULE MainModule",
        "PROC main()",
        "MoveJ",
        "tool0",
        "ENDPROC",
        "ENDMODULE"
    ]
    forbidden = ["ConfJ", "AccSet", "z0", "vmax"]
    return all(r in code for r in required) and not any(f in code for f in forbidden)


@app.post("/generate")
async def generate_code(request: TaskRequest):
    try:
        # Генерация промпта
        formatted_prompt = prompt.format(task_description=request.task_description)

        # Вызов модели
        raw_code = llm.invoke(formatted_prompt)

        # Очистка кода
        code = re.sub(r"```\w*", "", raw_code).strip()

        # Валидация
        if not validate_code(code):
            raise ValueError("Некорректный код: нарушены требования безопасности")

        return {"code": code}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
