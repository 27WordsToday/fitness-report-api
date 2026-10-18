import os
import re
import hmac
import json
import time
import httpx
from typing import Optional
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
MODEL = os.getenv("MODEL", "qwen2.5:3b")
API_KEY = os.getenv("API_KEY", "")

app = FastAPI(title="Fitness Report API")


class WeightEntry(BaseModel):
    date: str
    lb: float


class NutritionEntry(BaseModel):
    date: str
    calories: float
    protein_g: float = 0
    carbs_g: float = 0
    fat_g: float = 0


class WorkoutEntry(BaseModel):
    date: str
    type: str
    minutes: float


class Goals(BaseModel):
    calories: Optional[float] = None
    protein_g: Optional[float] = None
    workouts_per_week: Optional[int] = None
    target_weight_lb: Optional[float] = None


class Metrics(BaseModel):
    period_days: int = 7
    weight: list[WeightEntry] = []
    nutrition: list[NutritionEntry] = []
    workouts: list[WorkoutEntry] = []
    goals: Goals = Goals()


def compute_stats(m: Metrics) -> dict:
    stats = {"period_days": m.period_days}

    w = sorted(m.weight, key=lambda x: x.date)
    if w:
        stats["weight"] = {
            "start_lb": w[0].lb,
            "latest_lb": w[-1].lb,
            "change_lb": round(w[-1].lb - w[0].lb, 1),
        }
        if m.goals.target_weight_lb:
            stats["weight"]["lb_to_target"] = round(w[-1].lb - m.goals.target_weight_lb, 1)

    if m.nutrition:
        n = len(m.nutrition)
        stats["nutrition"] = {
            "days_logged": n,
            "avg_calories": round(sum(x.calories for x in m.nutrition) / n),
            "avg_protein_g": round(sum(x.protein_g for x in m.nutrition) / n),
            "avg_carbs_g": round(sum(x.carbs_g for x in m.nutrition) / n),
            "avg_fat_g": round(sum(x.fat_g for x in m.nutrition) / n),
        }
        if m.goals.calories:
            stats["nutrition"]["calorie_goal"] = m.goals.calories
        if m.goals.protein_g:
            stats["nutrition"]["protein_goal_g"] = m.goals.protein_g

    if m.workouts:
        types = {}
        for x in m.workouts:
            types[x.type] = types.get(x.type, 0) + 1
        stats["workouts"] = {
            "count": len(m.workouts),
            "total_minutes": round(sum(x.minutes for x in m.workouts)),
            "per_week": round(len(m.workouts) / m.period_days * 7, 1),
            "types": types,
        }
        if m.goals.workouts_per_week:
            stats["workouts"]["weekly_goal"] = m.goals.workouts_per_week

    return stats


def compute_findings(stats: dict) -> list:
    """The code judges what is a win, a gap, or info, and writes the tip for each gap.
    rank: lower number = more important."""
    f = []

    def add(kind, rank, text, tip=None):
        f.append({"kind": kind, "rank": rank, "text": text, "tip": tip})

    period = stats["period_days"]

    w = stats.get("weight")
    if w:
        ch = w["change_lb"]
        above = w.get("lb_to_target")
        if ch < 0:
            if above is None or above > 0:
                add("win", 1, f"Weight went down {abs(ch)} lb over the period, moving toward the target.")
            else:
                add("info", 9, f"Weight went down {abs(ch)} lb over the period.")
        elif ch > 0:
            if above is not None and above > 0:
                add("gap", 1, f"Weight went up {ch} lb over the period, moving away from the target.",
                    "Review your portions and daily activity to turn the scale back toward your target.")
            else:
                add("info", 9, f"Weight went up {ch} lb over the period.")
        else:
            add("info", 9, "Weight stayed the same over the period.")
        if above is not None:
            if above > 0:
                add("info", 9, f"Current weight is {above} lb above the target weight.")
            elif above < 0:
                add("info", 9, f"Current weight is {abs(above)} lb below the target weight.")
            else:
                add("win", 1, "Current weight is exactly at the target weight.")

    n = stats.get("nutrition")
    if n:
        if n["days_logged"] < period:
            add("gap", 3,
                f"Food was logged on only {n['days_logged']} of {period} days, so the averages may be incomplete.",
                "Log your food every day this week so your numbers are accurate.")
        goal = n.get("calorie_goal")
        if goal:
            avg = n["avg_calories"]
            diff = avg - goal
            if abs(diff) / goal <= 0.05:
                add("win", 3, f"Average calories ({avg}) were close to the goal ({round(goal)}).")
            elif diff > 0:
                add("gap", 2, f"Average calories ({avg}) were {round(diff)} above the goal ({round(goal)}).",
                    "Aim to bring your daily calories a little closer to your goal.")
            else:
                add("gap", 2, f"Average calories ({avg}) were {round(abs(diff))} below the goal ({round(goal)}).",
                    "Aim to bring your daily calories up closer to your goal so you can fuel your training.")
        pg = n.get("protein_goal_g")
        if pg:
            avg = n["avg_protein_g"]
            diff = avg - pg
            if diff >= 0:
                add("win", 2, f"Average protein ({avg} g) met the goal ({round(pg)} g).")
            else:
                short_pct = abs(diff) / pg
                rank = 1 if short_pct >= 0.10 else 4
                add("gap", rank,
                    f"Average protein ({avg} g) was {round(abs(diff))} g below the goal ({round(pg)} g).",
                    "Add a little more protein to each meal to close the gap to your goal.")

    wk = stats.get("workouts")
    if wk:
        goal = wk.get("weekly_goal")
        if goal:
            expected = goal * period / 7
            if wk["count"] >= expected - 0.01:
                add("win", 2, f"Completed {wk['count']} workouts, meeting the goal of {round(expected)} for this period.")
            else:
                short = max(1, round(expected - wk["count"]))
                plural = "workout" if short == 1 else "workouts"
                add("gap", 2,
                    f"Completed {wk['count']} workouts against a goal of {round(expected)} for this period, which is {short} short.",
                    f"Schedule {short} more {plural} this week to hit your goal.")
        else:
            add("info", 9, f"Completed {wk['count']} workouts.")
        add("info", 9, f"Workout types: {', '.join(wk['types'].keys())}. Total workout time was {wk['total_minutes']} minutes.")

    return f


def pick_top(findings: list, kind: str):
    items = [x for x in findings if x["kind"] == kind]
    if not items:
        return None
    return sorted(items, key=lambda x: x["rank"])[0]["text"]


def build_tips(findings: list) -> list:
    gaps = sorted([x for x in findings if x["kind"] == "gap" and x["tip"]], key=lambda x: x["rank"])
    tips = [g["tip"] for g in gaps][:3]
    if not tips:
        tips = ["You're on track. Keep up the same routine next week."]
    return tips


def build_prompt(findings: list, top_win, top_gap) -> str:
    infos = [x["text"] for x in findings if x["kind"] == "info"]
    info_lines = "\n".join(f"- {t}" for t in infos) if infos else "- (none)"

    if top_win and top_gap:
        shape = (
            "Sentence 1 states the good news. Sentence 2 states the main thing to improve."
        )
    elif top_gap:
        shape = (
            "There is no good news to report this period. Do NOT claim progress, "
            "momentum, or that they are on track. Sentence 1 plainly states the main "
            "thing to improve. Sentence 2 is a brief, neutral, encouraging line about "
            "the week ahead that makes no claim about results."
        )
    elif top_win:
        shape = (
            "Sentence 1 states the good news. Sentence 2 is a short line about "
            "keeping it up."
        )
    else:
        shape = (
            "Sentence 1 says there is not much data yet. Sentence 2 encourages "
            "logging more."
        )

    return (
        "You are a friendly, practical fitness coach writing a short report.\n"
        "The facts below are verified and already correct. Use ONLY these facts. "
        "Do not add numbers, change any fact, or mention anything not listed.\n\n"
        f"Good news: {top_win or 'none this period'}\n"
        f"Main thing to improve: {top_gap or 'none this period'}\n\n"
        f"Other info:\n{info_lines}\n\n"
        "Reply with JSON only, in exactly this shape:\n"
        '{"summary": "..."}\n\n'
        "Rules:\n"
        f"- summary: EXACTLY 2 sentences. {shape}\n"
        "- Write in second person, speaking directly to the reader as 'you'. "
        "Never say 'the participant' or 'the user'.\n"
        "- Never write the words 'top win', 'top gap', 'good news' or 'main thing' "
        "in the summary.\n"
        "- A shortfall is never progress. Do not spin a gap as a positive.\n"
        "- Do not give advice or tips. Only describe how the period went.\n"
        "- Do not give medical advice."
    )


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL}


@app.post("/report")
def report(metrics: Metrics, x_api_key: str = Header(default="")):
    if not API_KEY or not hmac.compare_digest(x_api_key, API_KEY):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

    started = time.time()
    stats = compute_stats(metrics)
    findings = compute_findings(stats)

    # No data at all: skip the model and return a plain message.
    if not findings:
        return {
            "stats": stats,
            "findings": [],
            "top_win": None,
            "top_gap": None,
            "summary": "There is no data for this period yet. Log your weight, meals and workouts to get a report.",
            "suggestions": ["Start by logging today's weight, meals and workouts."],
            "seconds": round(time.time() - started, 1),
        }

    top_win = pick_top(findings, "win")
    top_gap = pick_top(findings, "gap")
    tips = build_tips(findings)
    payload = {
        "model": MODEL,
        "prompt": build_prompt(findings, top_win, top_gap),
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.2,
            "num_predict": 120,
            "num_ctx": 2048,
        },
    }
    try:
        r = httpx.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=300)
        r.raise_for_status()
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Model unavailable: {e}")

    raw = r.json().get("response", "")
    try:
        summary = json.loads(raw).get("summary", "")
    except json.JSONDecodeError:
        summary = raw

    summary = re.sub(r"(?i)\btop\s+(win|gap)\b:?", "", summary)
    summary = " ".join(summary.split())

    return {
        "stats": stats,
        "findings": [x["text"] for x in findings],
        "top_win": top_win,
        "top_gap": top_gap,
        "summary": summary,
        "suggestions": tips,
        "seconds": round(time.time() - started, 1),
    }
