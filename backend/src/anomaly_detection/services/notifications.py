"""Notification service to fire alert webhooks to external incident responders."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

import httpx

from anomaly_detection.logging import get_logger

if TYPE_CHECKING:
    from anomaly_detection.config import Settings

logger = get_logger(__name__)


class NotificationService:
    """Orchestrates sending alert notifications to Slack, PagerDuty, or security webhooks."""

    def __init__(self, settings: Settings) -> None:
        self.webhook_url = settings.alert_webhook_url
        self.env = settings.environment

    async def notify_alert(
        self,
        alert_id: uuid.UUID,
        severity: str,
        suspected_attack_type: str | None,
        score: float,
        src_ip: str,
        dst_ip: str,
        protocol: int,
    ) -> None:
        """Post a webhook payload to an external security operations channel."""
        if not self.webhook_url:
            logger.debug(
                "alert_webhook_skipped",
                reason="no_webhook_configured",
                alert_id=str(alert_id),
            )
            return

        payload = {
            "text": f"🚨 *Anomaly Alert Raised: {severity.upper()}*",
            "blocks": [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"🚨 *Anomaly Alert Raised: {severity.upper()}*",
                    },
                },
                {
                    "type": "section",
                    "fields": [
                        {"type": "mrkdwn", "text": f"*Alert ID:*\n`{alert_id}`"},
                        {"type": "mrkdwn", "text": f"*Environment:*\n`{self.env}`"},
                        {
                            "type": "mrkdwn",
                            "text": f"*Suspected Threat:*\n`{suspected_attack_type or 'Unknown'}`",
                        },
                        {"type": "mrkdwn", "text": f"*Anomaly Score:*\n`{score:.4f}`"},
                        {"type": "mrkdwn", "text": f"*Source Connection:*\n`{src_ip}`"},
                        {"type": "mrkdwn", "text": f"*Destination Connection:*\n`{dst_ip}`"},
                    ],
                },
            ],
        }

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.post(self.webhook_url, json=payload)
                if response.status_code in (200, 201):
                    logger.info("alert_webhook_delivered", alert_id=str(alert_id))
                else:
                    logger.warning(
                        "alert_webhook_failed",
                        alert_id=str(alert_id),
                        status_code=response.status_code,
                        response=response.text[:200],
                    )
        except Exception as exc:
            logger.error(
                "alert_webhook_exception",
                alert_id=str(alert_id),
                error=str(exc),
            )
