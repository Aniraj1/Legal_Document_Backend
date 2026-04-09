import hashlib
import json
import logging
from datetime import datetime, timedelta, timezone

import requests
from django.conf import settings

logger = logging.getLogger(__name__)

ANALYTICS_KEY_PREFIX = "legal:analytics:user"
EVENTS_LIST_MAX = 2000
EVENTS_TTL_SECONDS = 30 * 24 * 60 * 60

WINDOW_SECONDS = {
    "24h": 24 * 60 * 60,
    "7d": 7 * 24 * 60 * 60,
    "30d": 30 * 24 * 60 * 60,
    "all": None,
}


def hash_query_text(query):
    normalized = (query or "").strip().lower()
    if not normalized:
        return ""
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def _redis_configured():
    return bool(
        getattr(settings, "UPSTASH_REDIS_REST_URL", "")
        and getattr(settings, "UPSTASH_REDIS_REST_TOKEN", "")
    )


def _events_key(user_id):
    return f"{ANALYTICS_KEY_PREFIX}:{user_id}:events"


def _registry_key(user_id):
    return f"{ANALYTICS_KEY_PREFIX}:{user_id}:keys"


def _pipeline(commands):
    if not _redis_configured():
        return []

    base_url = settings.UPSTASH_REDIS_REST_URL.rstrip("/")
    token = settings.UPSTASH_REDIS_REST_TOKEN
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(
            f"{base_url}/pipeline",
            headers=headers,
            json=commands,
            timeout=8,
        )
        response.raise_for_status()
        data = response.json()
        if isinstance(data, list):
            return data
        logger.error("[Analytics] Unexpected pipeline response: %s", data)
        return []
    except Exception as exc:
        logger.error("[Analytics] Redis pipeline failed: %s", exc)
        return []


def track_user_analytics_event(user_id, event_data):
    if not _redis_configured() or not user_id:
        return

    event_key = _events_key(user_id)
    registry_key = _registry_key(user_id)
    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **event_data,
    }
    serialized_event = json.dumps(event, separators=(",", ":"), default=str)

    commands = [
        ["LPUSH", event_key, serialized_event],
        ["LTRIM", event_key, 0, EVENTS_LIST_MAX - 1],
        ["EXPIRE", event_key, EVENTS_TTL_SECONDS],
        ["SADD", registry_key, event_key],
        ["EXPIRE", registry_key, EVENTS_TTL_SECONDS],
    ]
    _pipeline(commands)


def _parse_timestamp(raw_timestamp):
    if not raw_timestamp:
        return None
    try:
        normalized = raw_timestamp.replace("Z", "+00:00")
        dt = datetime.fromisoformat(normalized)
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def _get_user_events(user_id):
    if not _redis_configured() or not user_id:
        return []

    response = _pipeline([["LRANGE", _events_key(user_id), 0, EVENTS_LIST_MAX - 1]])
    if not response:
        return []

    raw_events = response[0].get("result", []) if isinstance(response[0], dict) else []
    parsed = []
    for raw in raw_events:
        try:
            if isinstance(raw, str):
                parsed.append(json.loads(raw))
            elif isinstance(raw, dict):
                parsed.append(raw)
        except Exception:
            continue
    return parsed


def _apply_window(events, window):
    seconds = WINDOW_SECONDS.get(window, WINDOW_SECONDS["7d"])
    if seconds is None:
        return events

    cutoff = datetime.now(timezone.utc) - timedelta(seconds=seconds)
    filtered = []
    for event in events:
        ts = _parse_timestamp(event.get("timestamp"))
        if ts is not None and ts >= cutoff:
            filtered.append(event)
    return filtered


def get_user_analytics_summary(user_id, window="7d"):
    events = _apply_window(_get_user_events(user_id), window)

    if not events:
        return {
            "totalQueries": 0,
            "totalEvents": 0,
            "totalUploads": 0,
            "totalDeletes": 0,
            "successCount": 0,
            "errorCount": 0,
            "successRate": 0,
            "avgTotalMs": 0,
            "avgVectorMs": 0,
            "avgGroqMs": 0,
            "topSourceTypes": [],
            "querySamples": [],
            "recentEvents": [],
            "hourlyDistribution": [{"hour": i, "count": 0} for i in range(24)],
            "topDocuments": [],
        }

    success_events = [event for event in events if event.get("status") == "success"]
    error_events = [event for event in events if event.get("status") == "error"]
    chat_events = [event for event in events if event.get("eventType") == "chat"]
    chat_success_events = [event for event in chat_events if event.get("status") == "success"]
    upload_events = [event for event in events if event.get("eventType") == "upload"]
    delete_events = [event for event in events if event.get("eventType") == "delete"]

    avg_total_ms = (
        sum(float(event.get("totalMs", 0)) for event in chat_success_events) / len(chat_success_events)
        if chat_success_events
        else 0
    )

    vector_events = [event for event in chat_success_events if event.get("vectorMs") is not None]
    avg_vector_ms = (
        sum(float(event.get("vectorMs", 0)) for event in vector_events) / len(vector_events)
        if vector_events
        else 0
    )

    groq_events = [event for event in chat_success_events if event.get("groqMs") is not None]
    avg_groq_ms = (
        sum(float(event.get("groqMs", 0)) for event in groq_events) / len(groq_events)
        if groq_events
        else 0
    )

    source_counts = {}
    for event in chat_success_events:
        for source_type in event.get("sourceLabels") or event.get("sourceTypes") or []:
            source_counts[source_type] = source_counts.get(source_type, 0) + 1

    top_source_types = sorted(
        [{"type": key, "count": value} for key, value in source_counts.items()],
        key=lambda item: item["count"],
        reverse=True,
    )[:10]

    query_counts = {}
    for event in chat_events:
        query_hash = event.get("queryHash")
        query_sample = event.get("querySample")
        if not query_hash or not query_sample:
            continue
        if query_hash not in query_counts:
            query_counts[query_hash] = {"query": query_sample, "count": 0}
        query_counts[query_hash]["count"] += 1

    query_samples = sorted(query_counts.values(), key=lambda item: item["count"], reverse=True)[:10]

    document_counts = {}
    for event in events:
        label = event.get("documentName") or event.get("documentId")
        if not label:
            continue
        document_counts[label] = document_counts.get(label, 0) + 1

    top_documents = sorted(
        [{"document": key, "count": value} for key, value in document_counts.items()],
        key=lambda item: item["count"],
        reverse=True,
    )[:10]

    hour_counts = {i: 0 for i in range(24)}
    for event in events:
        ts = _parse_timestamp(event.get("timestamp"))
        if ts is None:
            continue
        hour_counts[ts.hour] = hour_counts.get(ts.hour, 0) + 1

    hourly_distribution = [{"hour": hour, "count": hour_counts.get(hour, 0)} for hour in range(24)]

    return {
        "totalQueries": len(chat_events),
        "totalEvents": len(events),
        "totalUploads": len(upload_events),
        "totalDeletes": len(delete_events),
        "successCount": len(success_events),
        "errorCount": len(error_events),
        "successRate": (len(success_events) / len(events) * 100) if events else 0,
        "avgTotalMs": avg_total_ms,
        "avgVectorMs": avg_vector_ms,
        "avgGroqMs": avg_groq_ms,
        "topSourceTypes": top_source_types,
        "querySamples": query_samples,
        "recentEvents": events[:20],
        "hourlyDistribution": hourly_distribution,
        "topDocuments": top_documents,
    }


def clear_user_analytics(user_id):
    if not _redis_configured() or not user_id:
        return

    registry_key = _registry_key(user_id)
    members_resp = _pipeline([["SMEMBERS", registry_key]])
    keys = set()
    if members_resp and isinstance(members_resp[0], dict):
        for item in members_resp[0].get("result", []):
            if isinstance(item, str):
                keys.add(item)

    keys.add(_events_key(user_id))
    keys.add(registry_key)

    commands = [["DEL", key] for key in keys]
    if commands:
        _pipeline(commands)
