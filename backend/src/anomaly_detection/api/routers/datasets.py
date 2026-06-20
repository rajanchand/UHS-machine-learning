"""Datasets router — upload, list, preview, download, delete datasets."""

from __future__ import annotations

import os
import uuid
from typing import Any

from fastapi import APIRouter, File, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from sqlalchemy import select
from starlette.responses import Response

from anomaly_detection.db.models import Dataset
from anomaly_detection.schemas.common import DatasetResponse

router = APIRouter(prefix="/api/v1/datasets", tags=["datasets"])


@router.post("/upload", response_model=None)
async def upload_dataset(
    request: Request,
    file: UploadFile = File(...),
) -> Response | dict[str, Any]:
    """Upload a dataset CSV file."""
    settings = request.app.state.settings
    session_factory = request.app.state.session_factory

    if not file.filename or not file.filename.endswith(".csv"):
        return JSONResponse(status_code=400, content={"detail": "Only CSV files are supported"})

    # Save file
    upload_dir = str(settings.upload_dir)
    os.makedirs(upload_dir, exist_ok=True)
    file_id = str(uuid.uuid4())
    file_path = os.path.join(upload_dir, f"{file_id}_{file.filename}")

    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    # Analyze dataset
    try:
        import pandas as pd
        df = pd.read_csv(file_path)
        row_count = len(df)
        column_count = len(df.columns)
        columns = list(df.columns)
    except Exception as e:
        os.remove(file_path)
        return JSONResponse(status_code=400, content={"detail": f"Failed to parse CSV: {e!s}"})

    async with session_factory() as session:
        dataset = Dataset(
            name=file.filename.rsplit(".", 1)[0],
            filename=file.filename,
            file_path=file_path,
            file_size=len(content),
            row_count=row_count,
            column_count=column_count,
            columns_json=columns,
        )
        session.add(dataset)
        await session.commit()
        await session.refresh(dataset)

        return {
            "id": str(dataset.id),
            "name": dataset.name,
            "filename": dataset.filename,
            "file_size": dataset.file_size,
            "row_count": row_count,
            "column_count": column_count,
            "columns": columns,
            "message": "Dataset uploaded successfully",
        }


@router.get("")
async def list_datasets(request: Request) -> list[dict[str, Any]]:
    """List all uploaded datasets."""
    session_factory = request.app.state.session_factory

    async with session_factory() as session:
        result = await session.execute(select(Dataset).order_by(Dataset.created_at.desc()))
        datasets = result.scalars().all()
        return [
            DatasetResponse(
                id=d.id,
                name=d.name,
                filename=d.filename,
                file_size=d.file_size,
                row_count=d.row_count,
                column_count=d.column_count,
                columns_json=d.columns_json,
                created_at=d.created_at,
            ).model_dump()
            for d in datasets
        ]


@router.get("/{dataset_id}", response_model=None)
async def get_dataset(request: Request, dataset_id: str) -> Response | dict[str, Any]:
    """Get dataset details with preview."""
    session_factory = request.app.state.session_factory

    async with session_factory() as session:
        result = await session.execute(select(Dataset).where(Dataset.id == dataset_id))
        dataset = result.scalar_one_or_none()
        if not dataset:
            return JSONResponse(status_code=404, content={"detail": "Dataset not found"})

        # Get preview (first 10 rows)
        preview = []
        try:
            import pandas as pd
            df = pd.read_csv(dataset.file_path, nrows=10)
            preview = df.to_dict(orient="records")
        except Exception:
            pass

        return {
            "id": str(dataset.id),
            "name": dataset.name,
            "filename": dataset.filename,
            "file_size": dataset.file_size,
            "row_count": dataset.row_count,
            "column_count": dataset.column_count,
            "columns": dataset.columns_json,
            "preview": preview,
        }


@router.get("/{dataset_id}/download")
async def download_dataset(request: Request, dataset_id: str) -> Response:
    """Download a dataset file."""
    session_factory = request.app.state.session_factory

    async with session_factory() as session:
        result = await session.execute(select(Dataset).where(Dataset.id == dataset_id))
        dataset = result.scalar_one_or_none()
        if not dataset:
            return JSONResponse(status_code=404, content={"detail": "Dataset not found"})

        if not os.path.exists(dataset.file_path):
            return JSONResponse(status_code=404, content={"detail": "File not found"})

        return FileResponse(
            dataset.file_path,
            filename=dataset.filename,
            media_type="text/csv",
        )


@router.delete("/{dataset_id}", response_model=None)
async def delete_dataset(request: Request, dataset_id: str) -> Response | dict[str, Any]:
    """Delete a dataset."""
    session_factory = request.app.state.session_factory

    async with session_factory() as session:
        result = await session.execute(select(Dataset).where(Dataset.id == dataset_id))
        dataset = result.scalar_one_or_none()
        if not dataset:
            return JSONResponse(status_code=404, content={"detail": "Dataset not found"})

        if os.path.exists(dataset.file_path):
            os.remove(dataset.file_path)

        await session.delete(dataset)
        await session.commit()

    return {"message": "Dataset deleted"}
