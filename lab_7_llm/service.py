"""
Web service for model inference.
"""

# pylint: disable=too-few-public-methods
from pathlib import Path
from typing import Tuple

import pandas as pd
import torch
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic.dataclasses import dataclass as pydantic_dataclass

from lab_7_llm.main import LLMPipeline, TaskDataset


def init_application() -> Tuple[FastAPI, LLMPipeline]:
    """
    Initialize core application.

    Returns:
        tuple: tuple of two objects, instance of FastAPI server and
               LLMPipeline pipeline.
    """
    model_name = "Babelscape/wikineural-multilingual-ner"
    max_length = 120
    batch_size = 64
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dummy_data = pd.DataFrame({"source": [["dummy"]], "target": [[0]]})
    dataset = TaskDataset(dummy_data)

    pipeline_instance = LLMPipeline(
        model_name=model_name,
        dataset=dataset,
        max_length=max_length,
        batch_size=batch_size,
        device=device,
    )

    app_instance = FastAPI(title="NER Service")

    assets_path_instance = Path(__file__).parent / "assets"
    assets_path_instance.mkdir(exist_ok=True)

    app_instance.mount("/assets", StaticFiles(directory=str(assets_path_instance)), name="assets")

    return app_instance, pipeline_instance


app, pipeline = init_application()
assets_path = Path(__file__).parent / "assets"
templates = Jinja2Templates(directory=str(assets_path))


@pydantic_dataclass
class Query:
    """Query model for inference."""

    question: str


@app.get("/", response_class=HTMLResponse)
async def root(request: Request) -> HTMLResponse:
    """
    Root endpoint serving the main page.
    """
    return templates.TemplateResponse("index.html", {"request": request, "title": "NER Service"})


@app.post("/infer")
async def infer(query: Query) -> dict:
    """
    Main endpoint for model inference.
    """
    try:
        tokens = query.question.split()
        sample = (tokens, "0")

        prediction_str = pipeline.infer_sample(sample)

        if not prediction_str or prediction_str == "None":
            return {"infer": "No entities found"}

        if not isinstance(prediction_str, str):
            return {"infer": "Invalid prediction format"}

        pred_list = []
        if prediction_str.startswith("[") and prediction_str.endswith("]"):
            content = prediction_str[1:-1].strip()
            if content:
                pred_list = [int(x.strip()) for x in content.split(",") if x.strip()]

        id2label = pipeline._model.config.id2label if hasattr(pipeline._model, 'config') else {}
        if not id2label and pred_list:
            max_id = max(pred_list)
            id2label = {i: str(i) for i in range(max_id + 1)}

        result_parts = []
        for token, pred_id in zip(tokens, pred_list):
            label = id2label.get(pred_id, str(pred_id))
            if label == "O" or pred_id == 0:
                result_parts.append(token)
            else:
                if label.startswith("B-") or label.startswith("I-"):
                    label = label[2:]
                result_parts.append(f"{token}({label})")

        if len(pred_list) < len(tokens):
            result_parts.extend(tokens[len(pred_list):])

        result_text = " ".join(result_parts)

        return {"infer": result_text}

    except (ValueError, AttributeError, KeyError) as e:
        return {"infer": f"Error: {str(e)}"}
    except Exception as e:
        return {"infer": f"Unexpected error: {str(e)}"}
