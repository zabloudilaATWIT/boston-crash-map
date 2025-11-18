import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

dotenv_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set in .env")

print("Using API key prefix:", OPENAI_API_KEY[:10])  # for debugging

client = OpenAI(api_key=OPENAI_API_KEY)

DATA_PATH = Path(__file__).resolve().parent / "Crashes.csv"
print("Looking for CSV here:", DATA_PATH)
print("CSV exists:", DATA_PATH.exists())


try:
    crashes_df = pd.read_csv(DATA_PATH)
    print(f"Loaded dataset with {len(crashes_df)} rows from {DATA_PATH.name}")
except Exception as e:
    print(f"Warning: could not load {DATA_PATH}: {e}")
    crashes_df = None

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    answer: str


import re

def build_context_from_data(question: str) -> str:
    """
    Creates a compact crash data summary for the model using your columns:
    - crash_year
    - city_town_name
    - roadway
    - near_intersection_roadway
    - crash_hour / crash_time_num_rounded

    Also adds *generic* focus summaries for any city/road mentioned
    in the user's question, but only if they have a lot of crashes.
    """
    if crashes_df is None:
        return "Crash dataset could not be loaded."

    df = crashes_df.copy()
    parts = []

    # 1) Basic stats
    parts.append(f"Dataset size: {len(df)} crashes.")

    # 2) Crashes by year (short)
    if "crash_year" in df.columns:
        by_year = df["crash_year"].value_counts().sort_index()
        year_lines = [f"{int(year)}: {cnt}" for year, cnt in by_year.items()]
        parts.append("Crashes per year (first few): " + "; ".join(year_lines[:5]))

    # 3) Cities/towns (top 3)
    if "city_town_name" in df.columns:
        by_city = df["city_town_name"].value_counts(dropna=True)
        top_cities = by_city.head(3)
        city_lines = [f"{name}: {cnt}" for name, cnt in top_cities.items()]
        parts.append("Top cities/towns: " + "; ".join(city_lines))

    # 4) Major streets (top 3)
    if "roadway" in df.columns:
        by_road = df["roadway"].value_counts(dropna=True)
        top_roads = by_road.head(3)
        road_lines = [f"{name}: {cnt}" for name, cnt in top_roads.items()]
        parts.append("Highest crash streets: " + "; ".join(road_lines))

    # 5) Intersections (top 3) from roadway + near_intersection_roadway
    if "roadway" in df.columns and "near_intersection_roadway" in df.columns:
        tmp = df[["roadway", "near_intersection_roadway"]].dropna()
        tmp["intersection_label"] = (
            tmp["roadway"].str.strip()
            + " & "
            + tmp["near_intersection_roadway"].str.strip()
        )
        by_intersection = tmp["intersection_label"].value_counts()
        top_inters = by_intersection.head(3)
        if not top_inters.empty:
            inter_lines = [f"{name}: {cnt}" for name, cnt in top_inters.items()]
            parts.append(
                "Highest crash intersections: " + "; ".join(inter_lines)
            )

    # 6) Time-of-day patterns (top 3)
    time_col = None
    if "crash_hour" in df.columns:
        time_col = "crash_hour"
    elif "crash_time_num_rounded" in df.columns:
        time_col = "crash_time_num_rounded"

    if time_col:
        by_time = df[time_col].value_counts(dropna=True).head(3)
        time_lines = [f"{val}: {cnt}" for val, cnt in by_time.items()]
        parts.append(
            f"Peak crash times (from '{time_col}'): " + "; ".join(time_lines)
        )

    # 7) Generic focus area details based on the user's question
    q_lower = question.lower()
    focus_snippets = []
    HEAVY_THRESHOLD = 20  # only describe areas with >= this many crashes

    # 7a) If question mentions a city/town
    if "city_town_name" in df.columns:
        unique_cities = df["city_town_name"].dropna().unique()
        matching_cities = [
            city for city in unique_cities
            if str(city).lower() in q_lower
        ]
        for city in matching_cities[:2]:
            sub = df[df["city_town_name"] == city]
            if len(sub) < HEAVY_THRESHOLD:
                continue  # not a heavy crash area
            snippet = [f"In {city}, there are {len(sub)} crashes in the dataset."]
            if "roadway" in sub.columns:
                top_city_roads = (
                    sub["roadway"]
                    .value_counts(dropna=True)
                    .head(3)
                )
                roads_lines = [f"{r} ({c})" for r, c in top_city_roads.items()]
                if roads_lines:
                    snippet.append("Top crash streets there: " + "; ".join(roads_lines))
            focus_snippets.append(" ".join(snippet))

    # 7b) If question mentions a roadway name
    if "roadway" in df.columns:
        unique_roads = df["roadway"].dropna().unique()
        # sort by length (longer names first to avoid matching generic short words)
        unique_roads_sorted = sorted(unique_roads, key=lambda s: -len(str(s)))
        matching_roads = []
        for road in unique_roads_sorted:
            r_str = str(road)
            r_lower = r_str.lower()
            if len(r_lower) < 4:
                continue
            if r_lower in q_lower:
                matching_roads.append(r_str)
            if len(matching_roads) >= 2:  # don’t spam context
                break

        for road in matching_roads:
            sub = df[df["roadway"].str.contains(re.escape(road), case=False, na=False)]
            if len(sub) < HEAVY_THRESHOLD:
                continue  # only talk about it if it's a heavy crash area
            snippet = [f"Along {road}, there are {len(sub)} crashes in the dataset."]
            if "near_intersection_roadway" in sub.columns:
                top_cross = (
                    sub["near_intersection_roadway"]
                    .fillna("")
                    .str.strip()
                    .replace("", pd.NA)
                    .dropna()
                    .value_counts()
                    .head(3)
                )
                cross_lines = [f"{name} ({cnt})" for name, cnt in top_cross.items()]
                if cross_lines:
                    snippet.append("Top cross streets: " + "; ".join(cross_lines))
            focus_snippets.append(" ".join(snippet))

    if focus_snippets:
        parts.append(
            "Area-specific details based on the question:\n" +
            "\n".join(focus_snippets)
        )

    # Final context string
    return "\n".join(parts)[:2000]



# ------------------------------------------------------------
# 5. /chat endpoint
# ------------------------------------------------------------
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    question = req.message.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    data_context = build_context_from_data(question)

    try:
        resp = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an assistant for a senior project that analyzes "
                        "Boston crash data visualized on a map. Use the data "
                        "context when relevant. Give responses in 2-3 sentences or "
                        "about 3 bullet points."
                    ),
                },
                {
                    "role": "system",
                    "content": f"Data context:\n{data_context}",
                },
                {
                    "role": "user",
                    "content": question,
                },
            ],
        )
        answer = resp.choices[0].message.content
        return ChatResponse(answer=answer)

    except Exception as e:
        # show error for debugging
        print("Error in /chat:", repr(e))
        raise HTTPException(status_code=500, detail=str(e))
