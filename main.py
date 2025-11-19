import os

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

from fastapi.responses import FileResponse
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
HTML_PATH = BASE_DIR / "senior-proj-web.html" 

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not found in environmental variables")

client = OpenAI(api_key=OPENAI_API_KEY)

DATA_PATH = Path(__file__).resolve().parent / "Crashes.csv"


try:
    crashes_df = pd.read_csv(DATA_PATH)
    print(f"Loaded dataset with {len(crashes_df)} rows from {DATA_PATH.name}")
except Exception as e:
    print(f"Warning: could not load {DATA_PATH}: {e}")
    crashes_df = None

app = FastAPI()
@app.get("/")
async def serve_frontend():
    return FileResponse(HTML_PATH)


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
    Build a rich, compact crash-data summary to send as context to the model.

    Uses these columns when available:
      - crash_year, crash_date, crash_time, crash_hour, crash_time_num_rounded
      - city_town_name, crash_severity, crash_status
      - number_of_vehicles, speed_limit
      - light_conditions, weather_conditions, weather_new
      - manner_of_collision
      - roadway_junction_type, traffic_control_device_type, trafficway_description
      - roadway, near_intersection_roadway
      - number_of_travel_lanes_linked_rd, median_type_linked_rd
      - urban_type_linked_rd, urban_location_type_linked_rd

    It:
      - Summarizes global stats and patterns
      - Adds city/road-specific details if mentioned in the question
      - Adds simple comparisons when multiple locations are mentioned
      - Ends with explicit instructions for how the model should answer,
        including a strict limit of 3 sentences.

    The result is a plain-text summary to prepend to the model as context.
    """

    MAX_CONTEXT_CHARS = 3500  # allow a lot of info but still keep it reasonable

    global crashes_df
    if crashes_df is None:
        return (
            "Crash dataset could not be loaded. "
            "Answer very generally and say that no local crash data is available."
        )

    df = crashes_df.copy()
    parts: list[str] = []

    # ------------------------------------------------------------------
    # 1) BASIC DATASET SIZE + DATE RANGE
    # ------------------------------------------------------------------
    parts.append(f"Dataset size: {len(df)} crashes.")

    if "crash_date" in df.columns:
        try:
            dates = pd.to_datetime(df["crash_date"], errors="coerce")
            dates = dates.dropna()
            if not dates.empty:
                start_date = dates.min().date()
                end_date   = dates.max().date()
                parts.append(
                    f"Crash dates span from {start_date} to {end_date}."
                )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # 2) CRASHES BY YEAR + SIMPLE TREND
    # ------------------------------------------------------------------
    if "crash_year" in df.columns:
        by_year = df["crash_year"].value_counts(dropna=True).sort_index()
        if not by_year.empty:
            year_lines = [f"{int(y)}: {c}" for y, c in by_year.items()]
            parts.append("Crashes per year (first few): " + "; ".join(year_lines[:6]))

            # simple trend
            if len(by_year) >= 2:
                first_year = int(by_year.index[0])
                last_year  = int(by_year.index[-1])
                delta      = int(by_year.iloc[-1] - by_year.iloc[0])
                if delta > 0:
                    trend = "increased"
                elif delta < 0:
                    trend = "decreased"
                else:
                    trend = "stayed about the same"
                parts.append(
                    f"Trend: annual crashes have {trend} by {abs(delta)} "
                    f"from {first_year} to {last_year}."
                )

    # ------------------------------------------------------------------
    # 3) CITY/TOWN SUMMARY
    # ------------------------------------------------------------------
    if "city_town_name" in df.columns:
        by_city = df["city_town_name"].value_counts(dropna=True)
        if not by_city.empty:
            top_cities = by_city.head(5)
            city_lines = [f"{name}: {cnt}" for name, cnt in top_cities.items()]
            parts.append("Top cities/towns by crash count: " + "; ".join(city_lines))

    # ------------------------------------------------------------------
    # 4) CRASH SEVERITY & STATUS
    # ------------------------------------------------------------------
    if "crash_severity" in df.columns:
        sev_counts = df["crash_severity"].value_counts(dropna=True).head(5)
        if not sev_counts.empty:
            sev_lines = [f"{s}: {c}" for s, c in sev_counts.items()]
            parts.append("Crash severity distribution (top categories): " +
                         "; ".join(sev_lines))

    if "crash_status" in df.columns:
        status_counts = df["crash_status"].value_counts(dropna=True).head(4)
        if not status_counts.empty:
            status_lines = [f"{s}: {c}" for s, c in status_counts.items()]
            parts.append("Crash status counts: " + "; ".join(status_lines))

    # ------------------------------------------------------------------
    # 5) VEHICLES & SPEED LIMITS
    # ------------------------------------------------------------------
    if "number_of_vehicles" in df.columns:
        try:
            num_veh = pd.to_numeric(df["number_of_vehicles"], errors="coerce").dropna()
            if not num_veh.empty:
                parts.append(
                    f"Average number of vehicles per crash: {num_veh.mean():.2f}."
                )
                top_veh_counts = num_veh.value_counts().head(4)
                veh_lines = [f"{int(v)} vehicles: {c} crashes"
                             for v, c in top_veh_counts.items()]
                parts.append("Vehicle-count breakdown (top): " + "; ".join(veh_lines))
        except Exception:
            pass

    if "speed_limit" in df.columns:
        try:
            spd = pd.to_numeric(df["speed_limit"], errors="coerce").dropna()
            if not spd.empty:
                parts.append(
                    f"Speed limits range from {int(spd.min())} to {int(spd.max())}."
                )
                top_limits = spd.value_counts().head(4)
                spd_lines = [f"{int(s)} mph: {c} crashes"
                             for s, c in top_limits.items()]
                parts.append("Most common posted speed limits (by crash count): " +
                             "; ".join(spd_lines))
        except Exception:
            pass

    # ------------------------------------------------------------------
    # 6) TIME-OF-DAY PATTERNS
    # ------------------------------------------------------------------
    time_col = None
    if "crash_hour" in df.columns:
        time_col = "crash_hour"
    elif "crash_time_num_rounded" in df.columns:
        time_col = "crash_time_num_rounded"

    if time_col:
        by_time = df[time_col].value_counts(dropna=True).head(5)
        if not by_time.empty:
            time_lines = [f"{val}: {cnt}" for val, cnt in by_time.items()]
            parts.append(
                f"Peak crash times (from '{time_col}'): " + "; ".join(time_lines)
            )

    # ------------------------------------------------------------------
    # 7) WEATHER & LIGHT CONDITIONS
    # ------------------------------------------------------------------
    # Prefer weather_new if it exists, else weather_conditions
    weather_col = None
    if "weather_new" in df.columns:
        weather_col = "weather_new"
    elif "weather_conditions" in df.columns:
        weather_col = "weather_conditions"

    if weather_col:
        w_counts = df[weather_col].value_counts(dropna=True).head(4)
        if not w_counts.empty:
            w_lines = [f"{w}: {c}" for w, c in w_counts.items()]
            parts.append(
                f"Weather conditions with most crashes (from '{weather_col}'): "
                + "; ".join(w_lines)
            )

    if "light_conditions" in df.columns:
        light_counts = df["light_conditions"].value_counts(dropna=True).head(4)
        if not light_counts.empty:
            light_lines = [f"{l}: {c}" for l, c in light_counts.items()]
            parts.append(
                "Lighting conditions with most crashes: " + "; ".join(light_lines)
            )

    # ------------------------------------------------------------------
    # 8) COLLISION & ROAD CONFIGURATION TYPES
    # ------------------------------------------------------------------
    if "manner_of_collision" in df.columns:
        man_counts = df["manner_of_collision"].value_counts(dropna=True).head(4)
        if not man_counts.empty:
            man_lines = [f"{m}: {c}" for m, c in man_counts.items()]
            parts.append(
                "Most common manners of collision: " + "; ".join(man_lines)
            )

    if "roadway_junction_type" in df.columns:
        j_counts = df["roadway_junction_type"].value_counts(dropna=True).head(3)
        if not j_counts.empty:
            j_lines = [f"{j}: {c}" for j, c in j_counts.items()]
            parts.append(
                "Common roadway junction types at crashes: " + "; ".join(j_lines)
            )

    if "traffic_control_device_type" in df.columns:
        t_counts = df["traffic_control_device_type"].value_counts(dropna=True).head(3)
        if not t_counts.empty:
            t_lines = [f"{t}: {c}" for t, c in t_counts.items()]
            parts.append(
                "Common traffic control devices at crash locations: " +
                "; ".join(t_lines)
            )

    if "trafficway_description" in df.columns:
        tw_counts = df["trafficway_description"].value_counts(dropna=True).head(3)
        if not tw_counts.empty:
            tw_lines = [f"{d}: {c}" for d, c in tw_counts.items()]
            parts.append(
                "Trafficway descriptions most often involved: " +
                "; ".join(tw_lines)
            )

    # Linked road characteristics
    if "number_of_travel_lanes_linked_rd" in df.columns:
        try:
            lanes = pd.to_numeric(
                df["number_of_travel_lanes_linked_rd"], errors="coerce"
            ).dropna()
            if not lanes.empty:
                lanes_counts = lanes.value_counts().head(3)
                lanes_lines = [f"{int(n)} lanes: {c} crashes"
                               for n, c in lanes_counts.items()]
                parts.append(
                    "Typical number of travel lanes where crashes occur: " +
                    "; ".join(lanes_lines)
                )
        except Exception:
            pass

    if "median_type_linked_rd" in df.columns:
        med_counts = df["median_type_linked_rd"].value_counts(dropna=True).head(3)
        if not med_counts.empty:
            med_lines = [f"{m}: {c}" for m, c in med_counts.items()]
            parts.append(
                "Common median types at crash sites: " + "; ".join(med_lines)
            )

    if "urban_type_linked_rd" in df.columns:
        ut_counts = df["urban_type_linked_rd"].value_counts(dropna=True).head(3)
        if not ut_counts.empty:
            ut_lines = [f"{u}: {c}" for u, c in ut_counts.items()]
            parts.append(
                "Urban road types most represented: " + "; ".join(ut_lines)
            )

    if "urban_location_type_linked_rd" in df.columns:
        ul_counts = df["urban_location_type_linked_rd"].value_counts(dropna=True).head(3)
        if not ul_counts.empty:
            ul_lines = [f"{u}: {c}" for u, c in ul_counts.items()]
            parts.append(
                "Urban location types at crash sites: " + "; ".join(ul_lines)
            )

    # ------------------------------------------------------------------
    # 9) STREETS & INTERSECTIONS
    # ------------------------------------------------------------------
    by_intersection = None

    if "roadway" in df.columns:
        by_road = df["roadway"].value_counts(dropna=True)
        if not by_road.empty:
            top_roads = by_road.head(5)
            road_lines = [f"{name}: {cnt}" for name, cnt in top_roads.items()]
            parts.append("Highest crash roadways: " + "; ".join(road_lines))

    if "roadway" in df.columns and "near_intersection_roadway" in df.columns:
        tmp = df[["roadway", "near_intersection_roadway"]].dropna()
        if not tmp.empty:
            tmp["intersection_label"] = (
                tmp["roadway"].astype(str).str.strip()
                + " & "
                + tmp["near_intersection_roadway"].astype(str).str.strip()
            )
            by_intersection = tmp["intersection_label"].value_counts()
            top_inters = by_intersection.head(4)
            if not top_inters.empty:
                inter_lines = [f"{n}: {c}" for n, c in top_inters.items()]
                parts.append(
                    "Highest crash intersections: " + "; ".join(inter_lines)
                )

    # ------------------------------------------------------------------
    # 10) QUESTION-AWARE FOCUS AREAS (CITY / ROAD)
    # ------------------------------------------------------------------
    q_lower = question.lower()
    focus_snippets: list[str] = []
    HEAVY_THRESHOLD = 20

    # Heuristic for "complex" question (more detailed comparisons)
    complex_question = (
        len(question) > 120
        or any(
            kw in q_lower
            for kw in ["compare", "difference", "trend", "change", "safer", "worse"]
        )
    )

    # 10a) CITY-FOCUSED
    if "city_town_name" in df.columns:
        city_series = df["city_town_name"].dropna().astype(str)
        unique_cities = city_series.unique()

        matching_cities = [
            c for c in unique_cities if c.lower() in q_lower
        ]

        # City comparison if multiple mentioned
        if complex_question and len(matching_cities) >= 2:
            comp_lines = []
            for c in matching_cities[:4]:
                sub = df[df["city_town_name"] == c]
                comp_lines.append(f"{c}: {len(sub)} crashes")
            if comp_lines:
                parts.append(
                    "City comparison based on the question: " +
                    "; ".join(comp_lines)
                )

        # Detailed snippets for up to 2 heavy cities
        for c in matching_cities[:2]:
            sub = df[df["city_town_name"] == c]
            if len(sub) < HEAVY_THRESHOLD:
                continue

            snippet = [f"In {c}, there are {len(sub)} crashes in the dataset."]

            # Severity mix in that city
            if "crash_severity" in sub.columns:
                city_sev = sub["crash_severity"].value_counts(dropna=True).head(3)
                if not city_sev.empty:
                    sev_lines = [f"{s} ({cnt})" for s, cnt in city_sev.items()]
                    snippet.append(
                        "Crash severity mix there: " + "; ".join(sev_lines)
                    )

            # Top streets in that city
            if "roadway" in sub.columns:
                city_roads = sub["roadway"].value_counts(dropna=True).head(3)
                if not city_roads.empty:
                    r_lines = [f"{r} ({cnt})" for r, cnt in city_roads.items()]
                    snippet.append("Top crash roadways there: " + "; ".join(r_lines))

            # Weather in that city
            if weather_col and weather_col in sub.columns:
                city_weather = sub[weather_col].value_counts(dropna=True).head(2)
                if not city_weather.empty:
                    w_lines = [f"{w} ({cnt})" for w, cnt in city_weather.items()]
                    snippet.append(
                        "Most common weather at crashes there: " +
                        "; ".join(w_lines)
                    )

            focus_snippets.append(" ".join(snippet))

    # 10b) ROAD-FOCUSED
    if "roadway" in df.columns:
        road_series = df["roadway"].dropna().astype(str)
        unique_roads = road_series.unique()

        unique_roads_sorted = sorted(unique_roads, key=lambda s: -len(str(s)))
        matching_roads: list[str] = []

        for road in unique_roads_sorted:
            r_low = road.lower()
            if len(r_low) < 4:
                continue
            if r_low in q_lower:
                matching_roads.append(road)
            if len(matching_roads) >= 2:
                break

        for road in matching_roads:
            sub = df[df["roadway"].str.contains(re.escape(road), case=False, na=False)]
            if len(sub) < HEAVY_THRESHOLD:
                continue

            snippet = [f"Along {road}, there are {len(sub)} crashes in the dataset."]

            # Common cross streets
            if "near_intersection_roadway" in sub.columns:
                cross = (
                    sub["near_intersection_roadway"]
                    .fillna("")
                    .astype(str)
                    .str.strip()
                    .replace("", pd.NA)
                    .dropna()
                    .value_counts()
                    .head(3)
                )
                if not cross.empty:
                    cross_lines = [f"{n} ({c})" for n, c in cross.items()]
                    snippet.append("Top cross streets: " + "; ".join(cross_lines))

            # Speed limits along that road
            if "speed_limit" in sub.columns:
                try:
                    spd_sub = pd.to_numeric(
                        sub["speed_limit"], errors="coerce"
                    ).dropna()
                    if not spd_sub.empty:
                        spd_counts = spd_sub.value_counts().head(3)
                        spd_lines = [f"{int(s)} mph ({c})"
                                     for s, c in spd_counts.items()]
                        snippet.append(
                            "Common posted speed limits: " + "; ".join(spd_lines)
                        )
                except Exception:
                    pass

            focus_snippets.append(" ".join(snippet))

    if focus_snippets:
        parts.append(
            "Area-specific details based on the question:\n" +
            "\n".join(focus_snippets)
        )

    # ------------------------------------------------------------------
    # 11) GUIDANCE FOR THE MODEL (HOW TO ANSWER)
    # ------------------------------------------------------------------
    parts.append(
        "Use only the statistics and patterns summarized above when answering. "
        "You may describe trends, compare locations, highlight risky conditions "
        "(like certain times, weather, or road types), and explain differences "
        "between areas. Always keep your final answer to the user to at most "
        "three sentences, written as clear prose."
    )

    # ------------------------------------------------------------------
    # FINAL CONTEXT STRING
    # ------------------------------------------------------------------
    context = "\n".join(parts)
    return context[:MAX_CONTEXT_CHARS]




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
        print("Error in /chat:", repr(e))
        raise HTTPException(status_code=500, detail=str(e))
