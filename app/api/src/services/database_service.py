from __future__ import annotations

import datetime
import uuid

from supabase import Client


class DatabaseService:
    def __init__(self, client: Client) -> None:
        self._client = client

    def save_analysis(
        self,
        user_id: str,
        image_url: str,
        annotated_image_url: str | None,
        result_payload: dict,
        plant_id: str | None = None,
    ) -> dict:
        trichome = result_payload.get("trichome_result", {})
        stigma = result_payload.get("stigma_result", {})
        record = {
            "id": str(uuid.uuid4()),
            "performed_by": user_id,
            "plant_id": plant_id,
            "image_url": image_url,
            "annotated_image_url": annotated_image_url,
            "trichome_distribution": trichome.get("distribution"),
            "stigma_ratios": {
                "green": stigma.get("avg_green_ratio"),
                "orange": stigma.get("avg_orange_ratio"),
            },
            "maturity_stage": result_payload.get("maturity_stage"),
            "recommendation": result_payload.get("recommendation"),
            "detections": {
                "trichomes": trichome.get("detections"),
                "stigmas": stigma.get("detections"),
            },
            "created_at": datetime.datetime.now(datetime.UTC).isoformat(),
        }
        response = self._client.table("analyses").insert(record).execute()
        return response.data[0]

    def get_analysis(self, analysis_id: str, user_id: str | None = None) -> dict | None:
        query = self._client.table("analyses").select("*").eq("id", analysis_id)
        if user_id is not None:
            query = query.eq("performed_by", user_id)
        response = query.execute()
        if response.data:
            return response.data[0]
        return None

    def list_analyses(self, user_id: str | None = None, limit: int = 20) -> list[dict]:
        query = self._client.table("analyses").select("*")
        if user_id is not None:
            query = query.eq("performed_by", user_id)
        return query.order("created_at", desc=True).limit(limit).execute().data

    def update_analysis(self, analysis_id: str, fields: dict) -> dict:
        response = (
            self._client.table("analyses")
            .update(fields)
            .eq("id", analysis_id)
            .execute()
        )
        return response.data[0]

    def save_corrections(
        self,
        analysis_id: str,
        corrections: dict,
        trichome_distribution: dict,
        stigma_ratios: dict,
        maturity_stage: str,
        recommendation: str,
        detections: dict | None = None,
    ) -> dict:
        fields = {
            "corrections": corrections,
            "trichome_distribution": trichome_distribution,
            "stigma_ratios": stigma_ratios,
            "maturity_stage": maturity_stage,
            "recommendation": recommendation,
        }
        if detections is not None:
            fields["detections"] = detections
        response = (
            self._client.table("analyses")
            .update(fields)
            .eq("id", analysis_id)
            .execute()
        )
        return response.data[0]

    def delete_analysis(self, analysis_id: str) -> None:
        self._client.table("analyses").delete().eq("id", analysis_id).execute()

    def get_plant(self, plant_id: str, user_id: str | None = None) -> dict | None:
        query = self._client.table("plants").select("*").eq("id", plant_id)
        if user_id is not None:
            query = query.eq("created_by", user_id)
        response = query.execute()
        if response.data:
            return response.data[0]
        return None

    def delete_plant(self, plant_id: str) -> None:
        self._client.table("analyses").delete().eq("plant_id", plant_id).execute()
        self._client.table("plants").delete().eq("id", plant_id).execute()

    def create_plant(
        self,
        name: str,
        metadata: dict,
        created_by: str | None = None,
    ) -> dict:
        record = {
            "id": str(uuid.uuid4()),
            "created_by": created_by,
            "name": name,
            "status": "active",
            "metadata": metadata,
            "created_at": datetime.datetime.now(datetime.UTC).isoformat(),
        }
        response = self._client.table("plants").insert(record).execute()
        return response.data[0]

    def list_plants(self, user_id: str | None = None, name: str | None = None) -> list[dict]:
        query = self._client.table("plants").select("*")
        if user_id is not None:
            query = query.eq("created_by", user_id)
        if name is not None:
            query = query.ilike("name", name)
        return query.order("created_at", desc=True).execute().data

    def link_plant_to_analysis(self, analysis_id: str, plant_id: str) -> None:
        self._client.table("analyses").update({"plant_id": plant_id}).eq("id", analysis_id).execute()

    def list_plant_analyses(self, plant_id: str, user_id: str | None = None) -> list[dict]:
        query = (
            self._client.table("analyses")
            .select("*")
            .eq("plant_id", plant_id)
        )
        if user_id is not None:
            query = query.eq("performed_by", user_id)
        return query.order("created_at", desc=True).execute().data
