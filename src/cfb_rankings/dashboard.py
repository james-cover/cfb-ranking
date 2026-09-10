from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from cfb_rankings.config import load_settings
from cfb_rankings.storage import read_csv

st.set_page_config(page_title="CFB Ranking Lab", page_icon="🏈", layout="wide")
settings = load_settings()


@st.cache_data(ttl=300)
def load_outputs() -> dict[str, pd.DataFrame]:
    return {
        "rankings": read_csv(settings.predictions_dir / "current_rankings.csv"),
        "games": read_csv(settings.predictions_dir / "upcoming_game_predictions.csv"),
        "ap_evidence": read_csv(settings.models_dir / "ap_model_evidence.csv"),
        "game_evidence": read_csv(settings.models_dir / "game_model_evidence.csv"),
        "importance": read_csv(settings.models_dir / "game_model_feature_importance.csv"),
        "audit": read_csv(settings.processed_dir / "data_audit.csv"),
    }


def rank_text(value: object) -> str:
    if pd.isna(value):
        return "—"
    return str(int(float(value)))


def favorite_text(row: pd.Series, margin_column: str) -> str:
    margin = row.get(margin_column)
    if pd.isna(margin):
        return "—"
    if abs(float(margin)) < 0.05:
        return "Pick'em"
    team = row["home_team"] if float(margin) > 0 else row["away_team"]
    return f"{team} by {abs(float(margin)):.1f}"


data = load_outputs()
st.title("College Football Ranking Lab")
st.caption("Official AP • Predicted AP • Opponent-adjusted ranking • Game margins")

if data["rankings"].empty:
    st.warning(
        "No predictions exist yet. Add the API key, run the historical bootstrap, train, and predict."
    )
    st.code(
        "cfb-rankings bootstrap --start-year 2014\n"
        "cfb-rankings audit\n"
        "cfb-rankings build-features\n"
        "cfb-rankings train\n"
        "cfb-rankings predict"
    )
    st.stop()

rankings_tab, games_tab, evidence_tab, data_tab = st.tabs(
    ["Rankings", "Upcoming Games", "Model Evidence", "Data Health"]
)

with rankings_tab:
    order_label = st.selectbox(
        "Order table by",
        ["Independent model", "Actual AP", "Predicted AP"],
    )
    top_count = st.slider("Teams shown", min_value=25, max_value=75, value=30, step=5)
    rankings = data["rankings"].copy()
    order_column = {
        "Independent model": "independent_rank",
        "Actual AP": "actual_ap_rank",
        "Predicted AP": "predicted_ap_rank",
    }[order_label]
    rankings = rankings.sort_values(order_column, na_position="last").head(top_count)
    rankings["Actual AP"] = rankings["actual_ap_rank"].map(rank_text)
    rankings["Predicted AP"] = rankings["predicted_ap_rank"].map(rank_text)
    rankings["Independent"] = rankings["independent_rank"].map(rank_text)
    rankings["Rating"] = rankings["model_rating"].round(2)
    rankings["SOS rating"] = rankings["sos_elo"].round(0)
    rankings["Actual vs predicted"] = np.where(
        rankings["actual_ap_rank"].notna() & rankings["predicted_ap_rank"].notna(),
        rankings["actual_ap_rank"] - rankings["predicted_ap_rank"],
        np.nan,
    )
    st.dataframe(
        rankings[
            [
                "team", "record", "Actual AP", "Predicted AP", "Independent",
                "Rating", "SOS rating", "ranked_wins", "bad_losses", "Actual vs predicted",
            ]
        ].rename(
            columns={
                "team": "Team",
                "record": "Record",
                "ranked_wins": "Ranked wins",
                "bad_losses": "Bad losses",
            }
        ),
        hide_index=True,
        use_container_width=True,
    )
    st.caption(
        "Independent rating is the average selected-model neutral-site margin against "
        "every other FBS team. It is not a manually weighted score."
    )

with games_tab:
    games = data["games"].copy()
    if games.empty:
        st.info("No upcoming games with sufficient team history are currently available.")
    else:
        games["Matchup"] = games["away_team"] + " at " + games["home_team"]
        games.loc[games["neutral_site"].fillna(False).astype(bool), "Matchup"] = (
            games["away_team"] + " vs " + games["home_team"] + " (neutral)"
        )
        games["Predicted score"] = (
            games["away_team"]
            + " "
            + games["predicted_away_score"].round().astype(int).astype(str)
            + " – "
            + games["home_team"]
            + " "
            + games["predicted_home_score"].round().astype(int).astype(str)
        )
        games["Model margin"] = games.apply(
            lambda row: favorite_text(row, "model_home_margin"), axis=1
        )
        games["Market margin"] = games.apply(
            lambda row: favorite_text(row, "market_home_margin"), axis=1
        )
        games["Model edge"] = games["model_edge_home"].abs().round(1)
        games["Home win %"] = (100 * games["home_win_probability"]).round(1)
        games["Cover lean %"] = (
            100
            * np.where(
                games["model_edge_home"] >= 0,
                games["cover_probability_home"],
                1 - games["cover_probability_home"],
            )
        ).round(1)
        st.dataframe(
            games[
                [
                    "start_date", "Matchup", "Predicted score", "Model margin",
                    "Market margin", "model_ats_lean", "Model edge", "Home win %",
                    "Cover lean %", "line_fetched_at",
                ]
            ].rename(
                columns={
                    "start_date": "Kickoff",
                    "model_ats_lean": "ATS lean",
                    "line_fetched_at": "Line retrieved",
                }
            ),
            hide_index=True,
            use_container_width=True,
        )
        st.caption(
            "Sportsbook spreads are excluded from the model. Edge is the difference between "
            "the independent prediction and the median current line; it is not a guarantee."
        )

with evidence_tab:
    st.subheader("AP model selection")
    if data["ap_evidence"].empty:
        st.info("Train the models to populate chronological validation evidence.")
    else:
        st.dataframe(data["ap_evidence"], hide_index=True, use_container_width=True)
        champion = data["ap_evidence"].iloc[0]["model"]
        st.success(f"Selected AP champion: {champion}")
    st.subheader("Independent model validation")
    if not data["game_evidence"].empty:
        st.dataframe(data["game_evidence"], hide_index=True, use_container_width=True)
    if not data["importance"].empty:
        st.bar_chart(data["importance"].head(15).set_index("feature")["importance"])

with data_tab:
    if data["audit"].empty:
        st.info("Run `cfb-rankings audit` to create the data-health report.")
    else:
        st.dataframe(data["audit"], hide_index=True, use_container_width=True)
